#!/usr/bin/env python3
"""Run the Member 7 Level 2 340-record evaluation.

The script consumes data/self_collected/level2_manifest.csv exactly as handed
off. It does not run preprocessing and does not drop duplicate or difficult
samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "self_collected"
OUT = ROOT / "results" / "level2_340_20260915"
NUM_CLASSES = 10
INPUT_SIZE = 784

sys.path.insert(0, str(ROOT / "scripts"))
import generate_references as ref  # noqa: E402


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_pgm_payload(path: Path) -> np.ndarray:
    data = path.read_bytes()
    header = b"P5\n28 28\n255\n"
    if not data.startswith(header) or len(data) != len(header) + INPUT_SIZE:
        raise ValueError(f"invalid PGM: {path}")
    return np.frombuffer(data[len(header):], dtype=np.uint8).reshape(1, 28, 28)


def read_fixed_payload(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.int64, ndmin=1)
    if values.size != INPUT_SIZE:
        raise ValueError(f"{path}: expected 784 fixed raw values, got {values.size}")
    return values.reshape(1, 28, 28)


def infer_fixed_raw(batch_raw: np.ndarray, weights: tuple[np.ndarray, ...]) -> np.ndarray:
    conv1, conv2, fc1, fc2, fc3 = weights
    x = batch_raw.astype(np.int64, copy=False)
    x = np.maximum(ref.conv_fixed(x, conv1), 0)
    x = ref.pool(x)
    x = np.maximum(ref.conv_fixed(x, conv2), 0)
    x = ref.pool(x).reshape(len(batch_raw), -1)
    x = ref.narrow(np.einsum("ni,oi->no", x, fc1, optimize=True, dtype=np.int64))
    x = np.maximum(x, 0)
    x = ref.narrow(np.einsum("ni,oi->no", x, fc2, optimize=True, dtype=np.int64))
    x = np.maximum(x, 0)
    return ref.narrow(np.einsum("ni,oi->no", x, fc3, optimize=True, dtype=np.int64))


def run_command(command: list[str], log_path: Path, cwd: Path = ROOT) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log.write(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")


def verify_inputs(log_path: Path) -> None:
    run_command([sys.executable, "-B", "scripts/verify_member7_inputs.py"], log_path)


def load_inputs(rows: list[dict[str, str]], mode: str) -> tuple[np.ndarray, np.ndarray]:
    pgm_field = f"{mode}_relative"
    fixed_field = f"fixed_{mode}_relative"
    pgm = np.stack([read_pgm_payload(DATA / row[pgm_field]) for row in rows], axis=0)
    raw = np.stack([read_fixed_payload(DATA / row[fixed_field]) for row in rows], axis=0)
    return pgm, raw


def result_rows(
    manifest_rows: list[dict[str, str]],
    preprocess: str,
    inference_mode: str,
    logits: np.ndarray,
    fixed: bool,
) -> list[dict[str, object]]:
    labels = np.array([int(row["label"]) for row in manifest_rows], dtype=np.int64)
    predictions = np.argmax(logits, axis=1)
    rows: list[dict[str, object]] = []
    prefix = "raw_logit" if fixed else "logit"
    for i, row in enumerate(manifest_rows):
        out: dict[str, object] = {
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "preprocess": preprocess,
            "inference_mode": inference_mode,
            "prediction": int(predictions[i]),
            "correct": int(predictions[i] == labels[i]),
        }
        for c in range(NUM_CLASSES):
            out[f"{prefix}{c}"] = int(logits[i, c]) if fixed else float(logits[i, c])
        rows.append(out)
    return rows


def write_result_file(path: Path, rows: list[dict[str, object]], fixed: bool) -> None:
    prefix = "raw_logit" if fixed else "logit"
    fields = ["sample_id", "label", "preprocess", "inference_mode", "prediction", "correct"]
    fields += [f"{prefix}{i}" for i in range(NUM_CLASSES)]
    write_csv(path, rows, fields)


def read_hls_logits(path: Path, manifest_rows: list[dict[str, str]]) -> np.ndarray:
    rows = read_csv(path)
    if [row["sample_id"] for row in rows] != [row["sample_id"] for row in manifest_rows]:
        raise ValueError(f"HLS output order mismatch: {path}")
    if [int(row["label"]) for row in rows] != [int(row["label"]) for row in manifest_rows]:
        raise ValueError(f"HLS label mismatch: {path}")
    return np.array([[int(row[f"raw_logit{i}"]) for i in range(NUM_CLASSES)] for row in rows], dtype=np.int64)


def summarize_group(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    correct = sum(int(row["correct"]) for row in rows)
    summary: dict[str, object] = {
        "group": name,
        "preprocess": rows[0]["preprocess"],
        "inference_mode": rows[0]["inference_mode"],
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
    }
    for label in range(NUM_CLASSES):
        class_rows = [row for row in rows if int(row["label"]) == label]
        class_correct = sum(int(row["correct"]) for row in class_rows)
        summary[f"class_{label}_total"] = len(class_rows)
        summary[f"class_{label}_correct"] = class_correct
        summary[f"class_{label}_accuracy"] = class_correct / len(class_rows)
    return summary


def write_class_stats(path: Path, all_rows: dict[str, list[dict[str, object]]]) -> None:
    rows: list[dict[str, object]] = []
    for group, records in all_rows.items():
        for label in range(NUM_CLASSES):
            class_rows = [row for row in records if int(row["label"]) == label]
            correct = sum(int(row["correct"]) for row in class_rows)
            rows.append({
                "group": group,
                "preprocess": records[0]["preprocess"],
                "inference_mode": records[0]["inference_mode"],
                "label": label,
                "correct": correct,
                "total": len(class_rows),
                "accuracy": correct / len(class_rows),
            })
    write_csv(path, rows, ["group", "preprocess", "inference_mode", "label", "correct", "total", "accuracy"])


def write_confusion(path: Path, rows: list[dict[str, object]]) -> None:
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for row in rows:
        matrix[int(row["label"]), int(row["prediction"])] += 1
    out_rows: list[dict[str, object]] = []
    for true_label in range(NUM_CLASSES):
        row: dict[str, object] = {"row_is": "true_label", "true_label": true_label}
        for pred in range(NUM_CLASSES):
            row[f"pred_{pred}"] = int(matrix[true_label, pred])
        out_rows.append(row)
    write_csv(path, out_rows, ["row_is", "true_label", *[f"pred_{i}" for i in range(NUM_CLASSES)]])


def compare_fixed_hls(
    mode: str,
    fixed_logits: np.ndarray,
    hls_logits: np.ndarray,
    manifest_rows: list[dict[str, str]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    diff = np.abs(fixed_logits - hls_logits)
    sample_mismatch = np.any(diff != 0, axis=1)
    summary = {
        "preprocess": mode,
        "samples": int(diff.shape[0]),
        "values": int(diff.size),
        "mismatched_samples": int(np.sum(sample_mismatch)),
        "mismatched_values": int(np.sum(diff != 0)),
        "max_abs_diff": int(np.max(diff)),
    }
    details: list[dict[str, object]] = []
    for i, has_mismatch in enumerate(sample_mismatch):
        if not has_mismatch:
            continue
        for c in range(NUM_CLASSES):
            if diff[i, c] != 0:
                details.append({
                    "preprocess": mode,
                    "sample_id": manifest_rows[i]["sample_id"],
                    "logit_index": c,
                    "python_fixed_raw": int(fixed_logits[i, c]),
                    "optimized_hls_raw": int(hls_logits[i, c]),
                    "abs_diff": int(diff[i, c]),
                })
    return summary, details


def build_hls_driver(args: argparse.Namespace, log_path: Path) -> Path:
    build_dir = ROOT / "build" / "level2_340"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe = build_dir / "member7_level2_optimized_hls.exe"
    command = [
        args.compiler,
        "-std=c++11",
        "-O2",
        "-DLENET_USE_FIXED",
        "-DLENET_ACC_INT",
        "-I" + str(Path(args.hls_include)),
        "-Iconfig",
        "-Ihls/conv",
        "-Ihls/operators",
        "-Ihls/buffer",
        "-Ihls/top",
        "tests/tb_level2_340.cpp",
        "optimization/lenet_fast.cpp",
        "hls/conv/conv2d_systolic.cpp",
        "-o",
        str(exe),
    ]
    run_command(command, log_path)
    return exe


def run_hls(exe: Path, mode: str, output: Path, log_path: Path) -> None:
    run_command([
        str(exe),
        "--manifest", str(DATA / "level2_manifest.csv"),
        "--data-root", str(DATA),
        "--weights", str(ROOT / "weights" / "fixed"),
        "--mode", mode,
        "--output", str(output),
    ], log_path)


def write_readme(path: Path, consistency: list[dict[str, object]]) -> None:
    text = """# Member 7 Level 2 340 Evaluation

Dataset: level2_340_20260915. The denominator is 340 for every group and 34 for every class.

Inputs are read only from data/self_collected/level2_manifest.csv. Simple/Full float runs consume the listed PGM bytes as pixel/255. Python fixed and optimized HLS consume the listed fixed raw text files directly as A12 signed I7/F5 activation codes with scale 1/32.

optimized_hls here means g++ C simulation of optimization/lenet_fast.cpp with LENET_USE_FIXED and LENET_ACC_INT. It is not RTL co-simulation.

Confusion matrix files use rows as true labels and columns as predicted labels.

Fixed-vs-HLS consistency summary:
"""
    for item in consistency:
        text += (
            f"\n- {item['preprocess']}: mismatched_samples={item['mismatched_samples']}, "
            f"mismatched_values={item['mismatched_values']}, max_abs_diff={item['max_abs_diff']}"
        )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--hls-include", default=r"F:\Vivado\Vivado\2018.3\include")
    parser.add_argument("--skip-hls", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "run.log"
    log_path.write_text(f"Member 7 Level 2 run started {datetime.now(timezone.utc).isoformat()}\n", encoding="utf-8")

    verify_inputs(log_path)
    manifest_rows = read_csv(DATA / "level2_manifest.csv")
    ids = [row["sample_id"] for row in manifest_rows]
    labels = [int(row["label"]) for row in manifest_rows]
    if len(ids) != 340 or len(set(ids)) != 340:
        raise ValueError("manifest must contain 340 unique sample IDs")
    if Counter(labels) != Counter({i: 34 for i in range(NUM_CLASSES)}):
        raise ValueError("manifest must contain 34 samples per class")

    float_weights = ref.load_weights(ROOT / "weights", fixed=False)
    fixed_weights = ref.load_weights(ROOT / "weights", fixed=True)

    all_result_rows: dict[str, list[dict[str, object]]] = {}
    fixed_logits_by_mode: dict[str, np.ndarray] = {}
    hls_logits_by_mode: dict[str, np.ndarray] = {}

    for mode in ("simple", "full"):
        pgm, fixed_raw = load_inputs(manifest_rows, mode)
        float_logits = ref.infer_float(pgm, float_weights)
        fixed_logits = infer_fixed_raw(fixed_raw, fixed_weights)
        fixed_logits_by_mode[mode] = fixed_logits

        group = f"{mode}_float"
        rows = result_rows(manifest_rows, mode, "float", float_logits, fixed=False)
        all_result_rows[group] = rows
        write_result_file(OUT / f"results_{group}.csv", rows, fixed=False)

        group = f"{mode}_fixed"
        rows = result_rows(manifest_rows, mode, "fixed", fixed_logits, fixed=True)
        all_result_rows[group] = rows
        write_result_file(OUT / f"results_{group}.csv", rows, fixed=True)

    if args.skip_hls:
        raise RuntimeError("--skip-hls was set; optimized_hls outputs required by README_MEMBER7.md were not produced")

    exe = build_hls_driver(args, log_path)
    for mode in ("simple", "full"):
        raw_hls_path = OUT / f"optimized_hls_raw_{mode}.csv"
        run_hls(exe, mode, raw_hls_path, log_path)
        hls_logits = read_hls_logits(raw_hls_path, manifest_rows)
        hls_logits_by_mode[mode] = hls_logits
        group = f"{mode}_optimized_hls"
        rows = result_rows(manifest_rows, mode, "optimized_hls", hls_logits, fixed=True)
        all_result_rows[group] = rows
        write_result_file(OUT / f"results_{group}.csv", rows, fixed=True)

    combined = [row for group in (
        "simple_float", "simple_fixed", "simple_optimized_hls",
        "full_float", "full_fixed", "full_optimized_hls",
    ) for row in all_result_rows[group]]
    combined_fields = [
        "sample_id", "label", "preprocess", "inference_mode", "prediction", "correct",
        *[f"logit{i}" for i in range(NUM_CLASSES)],
        *[f"raw_logit{i}" for i in range(NUM_CLASSES)],
    ]
    write_csv(OUT / "results_all_2040.csv", combined, combined_fields)

    summary_rows = [summarize_group(group, rows) for group, rows in all_result_rows.items()]
    summary_fields = ["group", "preprocess", "inference_mode", "total", "correct", "accuracy"]
    for label in range(NUM_CLASSES):
        summary_fields += [f"class_{label}_total", f"class_{label}_correct", f"class_{label}_accuracy"]
    write_csv(OUT / "summary.csv", summary_rows, summary_fields)
    write_class_stats(OUT / "class_stats.csv", all_result_rows)
    for group, rows in all_result_rows.items():
        write_confusion(OUT / f"confusion_{group}.csv", rows)

    consistency_rows: list[dict[str, object]] = []
    consistency_details: list[dict[str, object]] = []
    for mode in ("simple", "full"):
        summary, details = compare_fixed_hls(mode, fixed_logits_by_mode[mode], hls_logits_by_mode[mode], manifest_rows)
        consistency_rows.append(summary)
        consistency_details.extend(details)
    write_csv(OUT / "fixed_vs_optimized_hls_consistency.csv", consistency_rows,
              ["preprocess", "samples", "values", "mismatched_samples", "mismatched_values", "max_abs_diff"])
    write_csv(OUT / "fixed_vs_optimized_hls_mismatches.csv", consistency_details,
              ["preprocess", "sample_id", "logit_index", "python_fixed_raw", "optimized_hls_raw", "abs_diff"])

    write_readme(OUT / "README.md", consistency_rows)

    files = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "run_manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "level2_340_20260915",
        "sample_count": 340,
        "per_class_count": 34,
        "manifest": {
            "path": "data/self_collected/level2_manifest.csv",
            "sha256": sha256(DATA / "level2_manifest.csv"),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "tools": {
            "compiler": shutil.which(args.compiler) or args.compiler,
            "hls_include": args.hls_include,
            "optimized_hls_execution": "g++ C simulation of optimization/lenet_fast.cpp; not RTL co-simulation",
        },
        "models_and_weights": {
            "float_weights": "weights/float/*.weight.txt",
            "fixed_weights": "weights/fixed/*.weight.txt",
            "float_weight_sha256": {p.name: sha256(p) for p in sorted((ROOT / "weights" / "float").glob("*.txt"))},
            "fixed_weight_sha256": {p.name: sha256(p) for p in sorted((ROOT / "weights" / "fixed").glob("*.txt"))},
        },
        "fixed_config": json.loads((ROOT / "config" / "quant_config.json").read_text(encoding="utf-8")),
        "source_files": {
            "scripts/run_member7_level2_eval.py": sha256(ROOT / "scripts" / "run_member7_level2_eval.py"),
            "tests/tb_level2_340.cpp": sha256(ROOT / "tests" / "tb_level2_340.cpp"),
            "optimization/lenet_fast.cpp": sha256(ROOT / "optimization" / "lenet_fast.cpp"),
            "optimization/fast_kernels.h": sha256(ROOT / "optimization" / "fast_kernels.h"),
            "hls/conv/conv2d_systolic.cpp": sha256(ROOT / "hls" / "conv" / "conv2d_systolic.cpp"),
        },
        "outputs": files,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(OUT),
        "summary": summary_rows,
        "consistency": consistency_rows,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
