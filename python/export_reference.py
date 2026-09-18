"""Export fixed MNIST samples and every float intermediate for HLS alignment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from model import LeNet5, model_metadata


def project_root():
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/lenet_float.pt")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-dir", default="reference/float")
    args = parser.parse_args()
    root = project_root()
    ds = datasets.MNIST(root=str(root / "data"), train=False, download=False, transform=transforms.ToTensor())
    model = LeNet5().cpu()
    payload = torch.load(root / args.checkpoint, map_location="cpu")
    model.load_state_dict(payload.get("model", payload))
    model.eval()
    out = root / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    index_manifest = []
    with torch.no_grad():
        for index in args.indices:
            image, label = ds[index]
            outputs = model.forward_with_intermediates(image.unsqueeze(0))
            sample_dir = out / ("sample_{:05d}".format(index))
            sample_dir.mkdir(parents=True, exist_ok=True)
            for name, value in outputs.items():
                arr = value.squeeze(0).cpu().numpy().astype(np.float32)
                np.save(sample_dir / (name + ".npy"), arr)
                np.savetxt(sample_dir / (name + ".txt"), arr.reshape(-1), fmt="%.9g")
            logits = outputs["logits"].squeeze(0)
            predicted = int(logits.argmax().item())
            index_manifest.append({"index": index, "label": int(label), "predicted": predicted, "correct": predicted == int(label), "sample_dir": str(sample_dir.relative_to(root))})
    (out / "manifest.json").write_text(json.dumps({"network": model_metadata(), "samples": index_manifest}, indent=2), encoding="utf-8")
    print(json.dumps(index_manifest, indent=2))


if __name__ == "__main__":
    main()
