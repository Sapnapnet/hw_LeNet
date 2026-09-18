"""Integer Golden Model and a bounded, integer-exact FP64 acceleration backend.

Real value = signed raw integer * 2**(-fraction_bits).
Quantization: nearest, ties to even, then asymmetric signed saturation.
MAC products retain ALL fractional bits; quantization occurs once per Conv/FC.
"""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Format:
    bits: int
    integer_bits: int

    def __post_init__(self):
        if not (2 <= self.bits <= 52 and 1 <= self.integer_bits <= self.bits):
            raise ValueError("Require 2 <= bits <= 52 and 1 <= integer_bits <= bits")

    @property
    def fraction_bits(self):
        return self.bits - self.integer_bits

    @property
    def lo(self):
        return -(1 << (self.bits - 1))

    @property
    def hi(self):
        return (1 << (self.bits - 1)) - 1

    def as_dict(self):
        return dict(bits=self.bits, integer_bits=self.integer_bits,
                    fraction_bits=self.fraction_bits, signed=True,
                    scale=2.0 ** -self.fraction_bits)


def quantize(values, fmt):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("NaN/Inf cannot be quantized")
    rounded = np.rint(values * (2.0 ** fmt.fraction_bits))
    count = int(np.count_nonzero((rounded < fmt.lo) | (rounded > fmt.hi)))
    return np.clip(rounded, fmt.lo, fmt.hi).astype(np.int64), count


def dequantize(raw, fmt):
    return np.asarray(raw, dtype=np.float64) * (2.0 ** -fmt.fraction_bits)


def round_shift(raw, shift):
    """Exact signed right shift with ties-to-even; floor division handles negatives."""
    raw = np.asarray(raw, dtype=np.int64)
    if shift < 0 or shift > 52:
        raise ValueError("Expected right shift in [0, 52]")
    if shift == 0:
        return raw.copy()
    divisor = 1 << shift
    base = raw // divisor
    remainder = raw % divisor
    up = (remainder > divisor // 2) | ((remainder == divisor // 2) & ((base & 1) != 0))
    return base + up.astype(np.int64)


def requantize(raw, shift, fmt):
    rounded = round_shift(raw, shift)
    count = int(np.count_nonzero((rounded < fmt.lo) | (rounded > fmt.hi)))
    return np.clip(rounded, fmt.lo, fmt.hi), count


def conv_int(x, weight):
    """Valid stride-1 cross correlation, OIHW weights, CHW features, int64 MAC."""
    if x.dtype != np.int64 or weight.dtype != np.int64:
        raise TypeError("conv_int requires int64")
    n, channels, h, w = x.shape
    out_channels, wc, kh, kw = weight.shape
    if channels != wc or h < kh or w < kw:
        raise ValueError("Invalid convolution dimensions")
    patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
    patches = patches.transpose(0, 2, 3, 1, 4, 5).reshape(-1, channels * kh * kw)
    acc = patches @ weight.reshape(out_channels, -1).T
    return acc.reshape(n, h - kh + 1, w - kw + 1, out_channels).transpose(0, 3, 1, 2)


def pool_int(x):
    n, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("This network requires even pooling dimensions")
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def formats(config):
    return {name: Format(value['bits'], value['integer_bits'])
            for name, value in config['formats'].items()}


def validate_config(config):
    fs = formats(config)
    a, w, acc = fs['activation'], fs['weight'], fs['accumulator']
    if config['rounding'] != 'nearest_ties_even' or config['overflow'] != 'saturation':
        raise ValueError("Unsupported rounding/overflow policy")
    if fs['output'] != a:
        raise ValueError("v1 uses the same activation and logits format")
    if acc.fraction_bits != a.fraction_bits + w.fraction_bits:
        raise ValueError("Accumulator must retain all product fractional bits")
    # 256 terms is the maximum fan-in (FC1). Use conservative 8 guard bits.
    if acc.bits < a.bits + w.bits + 8:
        raise ValueError("Accumulator needs full product width plus 8 guard bits")
    bound = 256 * (1 << (a.bits - 1)) * (1 << (w.bits - 1))
    if bound >= 1 << 62:
        raise ValueError("int64 implementation bound exceeded")
    return fs, bound


class IntegerLeNet:
    """Canonical reference: only int64 arithmetic after initial input/weight conversion."""
    def __init__(self, state, config):
        self.fs, self.bound = validate_config(config)
        expected = {'conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight', 'fc3.weight'}
        if set(state) != expected:
            raise ValueError("Expected exactly five no-bias tensors; do not read stale bias files")
        self.weights = {}
        self.weight_saturations = {}
        for name, value in state.items():
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            self.weights[name], self.weight_saturations[name] = quantize(value, self.fs['weight'])

    def forward(self, image):
        out = OrderedDict()
        stats = {}
        x, clipped = quantize(image, self.fs['activation'])
        if x.ndim != 4 or x.shape[1:] != (1, 28, 28):
            raise ValueError("Expected N x 1 x 28 x 28")
        out['input'] = x
        stats['input'] = {'saturated': clipped, 'elements': x.size}
        for layer, relu, pool in [('conv1', 'relu1', 'pool1'), ('conv2', 'relu2', 'pool2'),
                                  ('fc1', 'relu3', None), ('fc2', 'relu4', None),
                                  ('fc3', None, None)]:
            weight = self.weights[layer + '.weight']
            acc = conv_int(x, weight) if layer.startswith('conv') else x @ weight.T
            # The accumulator width was proven sufficient for EVERY representable input.
            if np.any(acc < self.fs['accumulator'].lo) or np.any(acc > self.fs['accumulator'].hi):
                raise AssertionError("Accumulator bound violated")
            x, clipped = requantize(acc, self.fs['weight'].fraction_bits, self.fs['activation'])
            name = 'logits' if layer == 'fc3' else layer
            out[name] = x
            stats[name] = {'saturated': clipped, 'elements': x.size}
            if relu:
                x = np.maximum(x, 0)
                out[relu] = x
            if pool:
                x = pool_int(x)
                out[pool] = x
                if pool == 'pool2':
                    x = x.reshape(x.shape[0], -1)
                    out['flatten'] = x
        return out, stats


class ExactDoubleLeNet:
    """Speed backend, NOT float/fake-quant inference.

    FP64 stores raw integer codes. Every product and any partial sum are exact
    because sum(abs(products)) <= bound < 2**53. Division is by a power of two.
    torch.round uses ties-to-even. CPU int64 remains the canonical reference.
    """
    def __init__(self, integer_model, device='cpu'):
        import torch
        self.torch = torch
        self.fs = integer_model.fs
        if integer_model.bound >= 1 << 53:
            raise ValueError("Cannot guarantee FP64 integer exactness")
        self.device = device
        self.weights = {k: torch.as_tensor(v, dtype=torch.float64, device=device)
                        for k, v in integer_model.weights.items()}

    def forward(self, images, collect=False):
        import torch.nn.functional as F
        torch = self.torch
        a = self.fs['activation']
        x = torch.as_tensor(images, dtype=torch.float64, device=self.device)
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Non-finite input")
        stats = {}
        out = OrderedDict()

        def clip(value, name):
            rounded = torch.round(value)
            stats[name] = {'saturated': int(((rounded < a.lo) | (rounded > a.hi)).sum().item()),
                           'elements': rounded.numel()}
            return rounded.clamp(a.lo, a.hi)

        def save(name, value):
            if collect:
                out[name] = value.to(torch.int64).cpu().numpy()

        x = clip(x * (2.0 ** a.fraction_bits), 'input')
        save('input', x)
        for layer, relu, pool in [('conv1', 'relu1', 'pool1'), ('conv2', 'relu2', 'pool2'),
                                  ('fc1', 'relu3', None), ('fc2', 'relu4', None),
                                  ('fc3', None, None)]:
            weight = self.weights[layer + '.weight']
            acc = F.conv2d(x, weight) if layer.startswith('conv') else F.linear(x, weight)
            name = 'logits' if layer == 'fc3' else layer
            x = clip(acc / (2.0 ** self.fs['weight'].fraction_bits), name)
            save(name, x)
            if relu:
                x = torch.relu(x)
                save(relu, x)
            if pool:
                x = F.max_pool2d(x, 2)
                save(pool, x)
                if pool == 'pool2':
                    x = x.flatten(1)
                    save('flatten', x)
        if collect:
            return out, stats
        return x.to(torch.int64).cpu().numpy(), stats
