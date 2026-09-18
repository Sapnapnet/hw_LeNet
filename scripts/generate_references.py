#!/usr/bin/env python3
"""Generate Float and W12/A12 Fixed logits for the MNIST test split.

The reference intentionally consumes the same exported text weights used by
the HLS testbench.  This avoids silently comparing HLS against a different
checkpoint or a PyTorch state-dict conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


INPUT_SIZE = 28
NUM_CLASSES = 10
DATA_SCALE = 32
NARROW_SHIFT = 11
RAW_MIN = -2048
RAW_MAX = 2047


def read_u32_be(data: bytes, offset: int) -> tuple[int, int]:
    return int.from_bytes(data[offset : offset + 4], "big"), offset + 4


def load_idx_images(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic, offset = read_u32_be(data, 0)
    count, offset = read_u32_be(data, offset)
    rows, offset = read_u32_be(data, offset)
    cols, offset = read_u32_be(data, offset)
    if (magic, rows, cols) != (2051, 28, 28):
        raise ValueError(f"invalid image IDX header in {path}: {magic}/{rows}/{cols}")
    expected = offset + count * rows * cols
    if len(data) != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {len(data)}")
    return np.frombuffer(data, dtype=np.uint8, count=count * rows * cols, offset=offset).reshape(
        count, 1, rows, cols
    )


def load_idx_labels(path: Path) -> np.ndarray:
    data = path.read_bytes()
    magic, offset = read_u32_be(data, 0)
    count, offset = read_u32_be(data, offset)
    if magic != 2049 or len(data) != offset + count:
        raise ValueError(f"invalid label IDX file {path}")
    return np.frombuffer(data, dtype=np.uint8, count=count, offset=offset)


def load_text(path: Path, size: int, dtype) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=1)
    if values.size != size:
        raise ValueError(f"{path}: expected {size} values, got {values.size}")
    return values.reshape(-1)


def load_weights(root: Path, fixed: bool) -> tuple[np.ndarray, ...]:
    directory = root / ("fixed" if fixed else "float")
    dtype = np.int64 if fixed else np.float32
    sizes = (150, 2400, 120 * 256, 84 * 120, 10 * 84)
    names = ("conv1", "conv2", "fc1", "fc2", "fc3")
    values = [load_text(directory / f"{name}.weight.txt", size, dtype) for name, size in zip(names, sizes)]
    if fixed:
        return (
            values[0].reshape(6, 1, 5, 5),
            values[1].reshape(16, 6, 5, 5),
            values[2].reshape(120, 256),
            values[3].reshape(84, 120),
            values[4].reshape(10, 84),
        )
    return (
        values[0].reshape(6, 1, 5, 5),
        values[1].reshape(16, 6, 5, 5),
        values[2].reshape(120, 256),
        values[3].reshape(84, 120),
        values[4].reshape(10, 84),
    )


def conv_float(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    kh, kw = weights.shape[-2:]
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(-2, -1))
    # windows: [N,C,OH,OW,KH,KW], weights: [OC,C,KH,KW]
    return np.einsum("nchwij,ocij->nohw", windows, weights, optimize=True).astype(np.float32)


def narrow(acc: np.ndarray) -> np.ndarray:
    """Q16 -> signed 12-bit Q5, round-half-to-even then saturate."""
    base = np.floor_divide(acc, 1 << NARROW_SHIFT)
    remainder = acc - (base << NARROW_SHIFT)
    half = 1 << (NARROW_SHIFT - 1)
    round_up = (remainder > half) | ((remainder == half) & ((base & 1) != 0))
    return np.clip(base + round_up.astype(np.int64), RAW_MIN, RAW_MAX).astype(np.int64)


def conv_fixed(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    kh, kw = weights.shape[-2:]
    windows = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(-2, -1))
    acc = np.einsum("nchwij,ocij->nohw", windows, weights, optimize=True, dtype=np.int64)
    return narrow(acc)


def pool(x: np.ndarray) -> np.ndarray:
    return np.maximum.reduce((x[:, :, 0::2, 0::2], x[:, :, 0::2, 1::2],
                              x[:, :, 1::2, 0::2], x[:, :, 1::2, 1::2]))


def infer_float(batch: np.ndarray, weights: tuple[np.ndarray, ...]) -> np.ndarray:
    conv1, conv2, fc1, fc2, fc3 = weights
    x = batch.astype(np.float32) / np.float32(255.0)
    x = np.maximum(conv_float(x, conv1), 0.0)
    x = pool(x)
    x = np.maximum(conv_float(x, conv2), 0.0)
    x = pool(x).reshape(len(batch), -1)
    x = np.maximum(x @ fc1.T, 0.0)
    x = np.maximum(x @ fc2.T, 0.0)
    return (x @ fc3.T).astype(np.float32)


def input_to_raw(images: np.ndarray) -> np.ndarray:
    # 255 is odd, so pixel*32/255 can never be an exact half integer.  rint
    # still documents the required ties-to-even rule and is deterministic.
    return np.rint(images.astype(np.float64) * DATA_SCALE / 255.0).astype(np.int64)


def infer_fixed(batch: np.ndarray, weights: tuple[np.ndarray, ...]) -> np.ndarray:
    conv1, conv2, fc1, fc2, fc3 = weights
    x = input_to_raw(batch)
    x = np.maximum(conv_fixed(x, conv1), 0)
    x = pool(x)
    x = np.maximum(conv_fixed(x, conv2), 0)
    x = pool(x).reshape(len(batch), -1)
    x = narrow(np.einsum("ni,oi->no", x, fc1, optimize=True, dtype=np.int64))
    x = np.maximum(x, 0)
    x = narrow(np.einsum("ni,oi->no", x, fc2, optimize=True, dtype=np.int64))
    x = np.maximum(x, 0)
    return narrow(np.einsum("ni,oi->no", x, fc3, optimize=True, dtype=np.int64))


def write_logits(path: Path, labels: np.ndarray, logits: np.ndarray, fixed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        prefix = "raw_logit" if fixed else "logit"
        writer.writerow(["index", "label", "pred", *[f"{prefix}{i}" for i in range(NUM_CLASSES)]])
        for index, (label, row) in enumerate(zip(labels, logits)):
            writer.writerow([index, int(label), int(np.argmax(row)), *[int(v) if fixed else float(v) for v in row]])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/mnist/raw"))
    parser.add_argument("--weights-root", type=Path, default=Path("weights"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/accuracy"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()

    images = load_idx_images(args.dataset_root / "t10k-images-idx3-ubyte")
    labels = load_idx_labels(args.dataset_root / "t10k-labels-idx1-ubyte")
    if len(images) != len(labels) or len(images) < args.limit:
        raise ValueError(f"image/label count mismatch or fewer than {args.limit} samples")
    images, labels = images[: args.limit], labels[: args.limit]

    float_logits = np.empty((args.limit, NUM_CLASSES), dtype=np.float32)
    fixed_logits = np.empty((args.limit, NUM_CLASSES), dtype=np.int64)
    float_weights = load_weights(args.weights_root, fixed=False)
    fixed_weights = load_weights(args.weights_root, fixed=True)
    for start in range(0, args.limit, args.batch_size):
        stop = min(start + args.batch_size, args.limit)
        float_logits[start:stop] = infer_float(images[start:stop], float_weights)
        fixed_logits[start:stop] = infer_fixed(images[start:stop], fixed_weights)
        print(f"reference {stop}/{args.limit}", flush=True)

    write_logits(args.output_dir / "python_float_10000.csv", labels, float_logits, fixed=False)
    write_logits(args.output_dir / "python_fixed_10000.csv", labels, fixed_logits, fixed=True)
    manifest = {
        "samples": int(args.limit),
        "input_scale": "pixel / 255.0 for float; round(pixel * 32 / 255) for fixed",
        "fixed_format": "W12/A12, raw activation scale 1/32, narrow shift 11, ties-even, saturate",
        "weights_root": str(args.weights_root),
        "outputs": ["python_float_10000.csv", "python_fixed_10000.csv"],
    }
    (args.output_dir / "python_reference_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
