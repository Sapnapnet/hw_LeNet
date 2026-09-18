#!/usr/bin/env python3
"""Download and validate the canonical MNIST IDX test files.

The script deliberately keeps the original IDX files (and their gzip
archives) so that the exact bytes used by the HLS testbench are auditable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


FILES = {
    "train-images-idx3-ubyte": {
        "count": 60000,
        "rows": 28,
        "cols": 28,
        "magic": 2051,
    },
    "train-labels-idx1-ubyte": {
        "count": 60000,
        "rows": None,
        "cols": None,
        "magic": 2049,
    },
    "t10k-images-idx3-ubyte": {
        "count": 10000,
        "rows": 28,
        "cols": 28,
        "magic": 2051,
    },
    "t10k-labels-idx1-ubyte": {
        "count": 10000,
        "rows": None,
        "cols": None,
        "magic": 2049,
    },
}

# The Google storage mirror is stable and supports HTTPS.  The original
# Yann LeCun URL is kept as a fallback for environments where the mirror is
# unavailable.
BASE_URLS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_u32_be(stream) -> int:
    data = stream.read(4)
    if len(data) != 4:
        raise ValueError("truncated IDX header")
    return int.from_bytes(data, "big")


def validate_idx(path: Path, spec: dict) -> None:
    with path.open("rb") as stream:
        magic = read_u32_be(stream)
        count = read_u32_be(stream)
        if magic != spec["magic"] or count != spec["count"]:
            raise ValueError(
                f"{path.name}: expected magic/count "
                f"{spec['magic']}/{spec['count']}, got {magic}/{count}"
            )
        if spec["rows"] is not None:
            rows = read_u32_be(stream)
            cols = read_u32_be(stream)
            if (rows, cols) != (spec["rows"], spec["cols"]):
                raise ValueError(
                    f"{path.name}: expected shape 28x28, got {rows}x{cols}"
                )
        # Confirm that all expected payload bytes are present.  This catches
        # partial downloads without loading the complete file into memory.
        expected_header = 16 if spec["rows"] is not None else 8
        expected_size = expected_header + spec["count"] * (
            spec["rows"] * spec["cols"] if spec["rows"] is not None else 1
        )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path.name}: expected {expected_size} bytes, got {actual_size}"
            )


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "FPGA-LeNet MNIST downloader/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)


def ensure_file(name: str, raw_dir: Path) -> dict:
    spec = FILES[name]
    gz_path = raw_dir / f"{name}.gz"
    raw_path = raw_dir / name

    if not raw_path.exists():
        if not gz_path.exists():
            last_error = None
            for base in BASE_URLS:
                try:
                    print(f"Downloading {base}{name}.gz", flush=True)
                    download(base + f"{name}.gz", gz_path)
                    break
                except Exception as error:  # pragma: no cover - network dependent
                    last_error = error
                    if gz_path.exists():
                        gz_path.unlink()
            else:
                raise RuntimeError(f"unable to download {name}.gz: {last_error}")
        print(f"Extracting {gz_path.name}", flush=True)
        with gzip.open(gz_path, "rb") as source, raw_path.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)

    validate_idx(raw_path, spec)
    return {
        "name": name,
        "gzip": gz_path.name,
        "raw": raw_path.name,
        "bytes": raw_path.stat().st_size,
        "sha256": sha256(raw_path),
        **spec,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=None,
        help="directory for MNIST files (default: repo/data/mnist/raw)",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(args.output) if args.output else repo_root / "data" / "mnist" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": "MNIST",
        "split": "train + t10k",
        "source_urls": list(BASE_URLS),
        "files": [ensure_file(name, raw_dir) for name in FILES],
    }
    manifest_path = raw_dir.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MNIST ready: {raw_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
