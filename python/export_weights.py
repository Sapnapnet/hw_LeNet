"""Export the trained checkpoint into deterministic HLS-friendly text/NPY files."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model import LeNet5, model_metadata


def project_root():
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/lenet_float.pt")
    parser.add_argument("--output-dir", default="weights/exported")
    args = parser.parse_args()
    root = project_root()
    device = torch.device("cpu")
    model = LeNet5().to(device)
    payload = torch.load(root / args.checkpoint, map_location=device)
    model.load_state_dict(payload.get("model", payload))
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"network": model_metadata(), "tensors": {}}
    for name, tensor in model.state_dict().items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        np.save(out / (name + ".npy"), arr)
        np.savetxt(out / (name + ".txt"), arr.reshape(-1), fmt="%.9g")
        manifest["tensors"][name] = {"shape": list(arr.shape), "dtype": "float32", "layout": "row-major flattened in .txt"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Exported", len(manifest["tensors"]), "tensors to", out)


if __name__ == "__main__":
    main()
