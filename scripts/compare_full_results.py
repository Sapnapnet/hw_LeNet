#!/usr/bin/env python3
"""Compare the two HLS 10k CSVs with the Python golden CSVs.

The resulting JSON separates classification accuracy from numeric alignment.
In particular, the Python-only baseline is never presented as HLS accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path, fixed: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty result file: {path}")
    indexes = np.array([int(row["index"]) for row in rows], dtype=np.int64)
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    predictions = np.array([int(row["pred"]) for row in rows], dtype=np.int64)
    prefix = "raw_logit" if fixed else "logit"
    logits = np.array(
        [[int(row[f"{prefix}{i}"]) if fixed else float(row[f"{prefix}{i}"]) for i in range(10)] for row in rows]
    )
    return indexes, labels, predictions, logits


def check_alignment(actual, reference, name: str) -> None:
    if actual[0].shape != reference[0].shape or not np.array_equal(actual[0], reference[0]):
        raise ValueError(f"{name}: sample indexes differ")
    if not np.array_equal(actual[1], reference[1]):
        raise ValueError(f"{name}: labels differ between HLS and Python files")


def summarize_float(hls, reference) -> dict:
    diff = np.abs(hls[3] - reference[3])
    return {
        "mode": "float",
        "samples": int(len(hls[0])),
        "correct": int(np.sum(hls[2] == hls[1])),
        "accuracy": float(np.mean(hls[2] == hls[1])),
        "hls_vs_python_float_max_abs_logit_error": float(np.max(diff)),
        "hls_vs_python_float_mean_abs_logit_error": float(np.mean(diff)),
        "hls_vs_python_float_p99_abs_logit_error": float(np.quantile(diff, 0.99)),
        "hls_vs_python_float_logit_mismatch_gt_1e-5": int(np.sum(diff > 1e-5)),
        "hls_vs_python_float_prediction_mismatch": int(np.sum(hls[2] != reference[2])),
        "first_five_hls_predictions": [int(x) for x in hls[2][:5]],
        "first_five_labels": [int(x) for x in hls[1][:5]],
    }


def summarize_fixed(hls, reference) -> dict:
    diff = np.abs(hls[3] - reference[3])
    return {
        "mode": "fixed",
        "samples": int(len(hls[0])),
        "correct": int(np.sum(hls[2] == hls[1])),
        "accuracy": float(np.mean(hls[2] == hls[1])),
        "hls_vs_python_fixed_raw_mismatch": int(np.sum(diff != 0)),
        "hls_vs_python_fixed_raw_total_values": int(diff.size),
        "hls_vs_python_fixed_raw_max_abs_diff": int(np.max(diff)),
        "hls_vs_python_fixed_raw_mean_abs_diff": float(np.mean(diff)),
        "hls_vs_python_fixed_prediction_mismatch": int(np.sum(hls[2] != reference[2])),
        "first_five_hls_predictions": [int(x) for x in hls[2][:5]],
        "first_five_labels": [int(x) for x in hls[1][:5]],
    }


def write_per_sample(path: Path, float_hls, float_ref, fixed_hls, fixed_ref) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "index", "label", "float_hls_pred", "float_python_pred",
            "float_max_abs_logit_error", "fixed_hls_pred", "fixed_python_pred",
            "fixed_raw_mismatch", "fixed_max_abs_raw_diff",
        ])
        for i in range(len(float_hls[0])):
            float_diff = np.abs(float_hls[3][i] - float_ref[3][i])
            fixed_diff = np.abs(fixed_hls[3][i] - fixed_ref[3][i])
            writer.writerow([
                int(i), int(float_hls[1][i]), int(float_hls[2][i]), int(float_ref[2][i]),
                float(np.max(float_diff)), int(fixed_hls[2][i]), int(fixed_ref[2][i]),
                int(np.sum(fixed_diff != 0)), int(np.max(fixed_diff)),
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hls-float", type=Path, required=True)
    parser.add_argument("--hls-fixed", type=Path, required=True)
    parser.add_argument("--python-float", type=Path, required=True)
    parser.add_argument("--python-fixed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/accuracy"))
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--hls-version", default="unknown")
    parser.add_argument("--float-macros", default="none")
    parser.add_argument("--fixed-macros", default="LENET_USE_FIXED LENET_ACC_INT")
    args = parser.parse_args()

    hls_float = read_csv(args.hls_float, fixed=False)
    hls_fixed = read_csv(args.hls_fixed, fixed=True)
    python_float = read_csv(args.python_float, fixed=False)
    python_fixed = read_csv(args.python_fixed, fixed=True)
    check_alignment(hls_float, python_float, "float")
    check_alignment(hls_fixed, python_fixed, "fixed")
    if not np.array_equal(hls_float[0], hls_fixed[0]) or not np.array_equal(hls_float[1], hls_fixed[1]):
        raise ValueError("Float and Fixed HLS files do not use the same sample/label order")

    summary = {
        "dataset": "MNIST t10k",
        "samples": int(len(hls_float[0])),
        "source_commit": args.source_commit,
        "hls_version": args.hls_version,
        "float_compile_macros": args.float_macros,
        "fixed_compile_macros": args.fixed_macros,
        "python_baseline": {
            "float_correct": int(np.sum(python_float[2] == python_float[1])),
            "float_accuracy": float(np.mean(python_float[2] == python_float[1])),
            "fixed_correct": int(np.sum(python_fixed[2] == python_fixed[1])),
            "fixed_accuracy": float(np.mean(python_fixed[2] == python_fixed[1])),
            "note": "Python-only reference; not an HLS accuracy result",
        },
        "hls_float": summarize_float(hls_float, python_float),
        "hls_fixed": summarize_fixed(hls_fixed, python_fixed),
        "files": {
            "hls_float": str(args.hls_float),
            "hls_fixed": str(args.hls_fixed),
            "python_float": str(args.python_float),
            "python_fixed": str(args.python_fixed),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hls_accuracy_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_per_sample(args.output_dir / "hls_accuracy_per_sample.csv", hls_float, python_float, hls_fixed, python_fixed)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
