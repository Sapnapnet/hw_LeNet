#!/usr/bin/env python3
"""Write a provenance manifest for the completed full-set run."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/accuracy/run_manifest.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        commit = "unknown"

    result_dir = root / "results" / "accuracy"
    names = [
        "hls_float_hls_csim_10000.csv",
        "hls_fixed_hls_csim_10000.csv",
        "python_float_10000.csv",
        "python_fixed_10000.csv",
        "float_vivado_hls_csim_10000.log",
        "fixed_vivado_hls_csim_10000.log",
        "hls_accuracy_summary.json",
        "hls_accuracy_per_sample.csv",
    ]
    files = {}
    for name in names:
        path = result_dir / name
        if path.exists():
            files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}

    dataset_manifest = root / "data" / "mnist" / "manifest.json"
    dataset = json.loads(dataset_manifest.read_text(encoding="utf-8")) if dataset_manifest.exists() else {}
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "source": {
            "git_commit": commit,
            "canonical_tree": "hls/ + config/ + tests/",
            "checkpoint_sha256": "c263e8fbdb6194f9323cdb199588be9422eb0f7f3c21ee59d4b8452ad02e16cb",
        },
        "hls": {
            "tool": "Vivado HLS 2018.3 Build 2405991",
            "executable": "F:/Vivado/Vivado/2018.3/bin/vivado_hls.bat",
            "part": "xc7z020clg400-1",
            "clock_ns": 10,
            "float_macros": [],
            "fixed_macros": ["LENET_USE_FIXED", "LENET_ACC_INT"],
            "commands": {
                "float": "F:/Vivado/Vivado/2018.3/bin/vivado_hls.bat -f scripts/hls/csim_float.tcl",
                "fixed": "F:/Vivado/Vivado/2018.3/bin/vivado_hls.bat -f scripts/hls/csim_fixed.tcl",
            },
        },
        "portable_driver": {
            "compiler": "MSYS2 ucrt64 g++ 15.1",
            "source": "tests/tb_mnist_10k.cpp + hls/top/lenet_accelerator.cpp + hls/conv/conv2d_systolic.cpp",
        },
        "results": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
