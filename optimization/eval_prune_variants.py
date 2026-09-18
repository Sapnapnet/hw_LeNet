# -*- coding: utf-8 -*-
# Offline fixed-point evaluation behind the aggressive5b channel selection.
# Mirrors the C datapath exactly (verified: 3-sample logits identical to CSim)
# and writes results/channel_ablation.json for the report.
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import generate_references as g

OUT = ROOT / 'optimization/results'

imgs = g.load_idx_images(ROOT / 'data/mnist/raw/t10k-images-idx3-ubyte')
y = g.load_idx_labels(ROOT / 'data/mnist/raw/t10k-labels-idx1-ubyte')
c1, c2, f1, f2, f3 = g.load_weights(ROOT / 'weights', fixed=True)


def run_model(k1, k2):
    x = g.input_to_raw(imgs)
    x = np.maximum(g.conv_fixed(x, c1[k1]), 0)
    x = g.pool(x)
    x = np.maximum(g.conv_fixed(x, c2[k2][:, k1]), 0)
    x = g.pool(x).reshape(len(imgs), -1)
    f1p = f1.reshape(120, 16, 16)[:, k2, :].reshape(120, -1)
    h = np.maximum(g.narrow(x @ f1p.T), 0)
    z = np.maximum(g.narrow(h @ f2.T), 0)
    out = g.narrow(z @ f3.T)
    return float(np.mean(np.argmax(out, 1) == y))


K2 = [10, 0, 3, 2, 12, 8, 13, 6, 14, 1, 9, 5]   # FC1 weight-norm ranking, keep 12
K1_FINAL = [0, 1, 2, 4, 5]                     # drop channel 3

result = {'samples': int(len(imgs)), 'conv2_keep': K2, 'conv1_keep': K1_FINAL,
          'note': 'fixed-point numpy simulation, matches CSim bit-exactly on checked samples'}

result['baseline_6x16'] = run_model(list(range(6)), list(range(16)))
print('baseline 6x16:', result['baseline_6x16'], flush=True)

# Conv1 drop-one sweep with the final Conv2 keep set.
sweep = {}
for drop in range(6):
    k1 = [c for c in range(6) if c != drop]
    sweep[str(drop)] = {'keep': k1, 'accuracy': run_model(k1, K2)}
    print('drop conv1 ch%d keep=%s acc=%.4f' % (drop, k1, sweep[str(drop)]['accuracy']), flush=True)
result['conv1_drop_sweep_with_final_conv2'] = sweep

# Conv2 12-channel rankings conditioned on the final pruned Conv1 net.
x = g.input_to_raw(imgs)
p1 = g.pool(np.maximum(g.conv_fixed(x, c1[K1_FINAL]), 0))


def run_from_p1(k2):
    a2 = np.maximum(g.conv_fixed(p1, c2[k2][:, K1_FINAL]), 0)
    p2 = g.pool(a2).reshape(len(imgs), -1)
    f1p = f1.reshape(120, 16, 16)[:, k2, :].reshape(120, -1)
    h = np.maximum(g.narrow(p2 @ f1p.T), 0)
    z = np.maximum(g.narrow(h @ f2.T), 0)
    out = g.narrow(z @ f3.T)
    return float(np.mean(np.argmax(out, 1) == y))


a2f = np.maximum(g.conv_fixed(p1, c2[:, K1_FINAL]), 0)
p2f = g.pool(a2f)
rankings = {
    'fc1_weight_l1 (final)': (K2, np.sum(np.abs(f1.reshape(120, 16, 16)), axis=(0, 2))),
    'conv2_weight_l1': (None, np.sum(np.abs(c2[:, K1_FINAL]), axis=(1, 2, 3))),
    'activation_mean_abs': (None, np.mean(np.abs(p2f), axis=(0, 2, 3))),
    'activation_nonzero': (None, np.mean(p2f > 0, axis=(0, 2, 3))),
}
rank_out = {}
for name, (k2_fixed, s) in rankings.items():
    k2 = k2_fixed if k2_fixed is not None else np.argsort(s)[::-1][:12].tolist()
    rank_out[name] = {'keep': k2, 'accuracy': run_from_p1(k2)}
    print(name, 'keep12=', k2, 'acc=%.4f' % rank_out[name]['accuracy'], flush=True)
result['conv2_keep12_rankings_with_final_conv1'] = rank_out

# Ladder accuracies for the intermediate variants.
result['aggressive12b_6x12'] = run_model(list(range(6)), K2)
print('aggressive12b 6x12:', result['aggressive12b_6x12'], flush=True)
result['final_aggressive5b_5x12'] = run_model(K1_FINAL, K2)
print('final aggressive5b 5x12:', result['final_aggressive5b_5x12'], flush=True)

(OUT / 'channel_ablation.json').write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('written', OUT / 'channel_ablation.json')
