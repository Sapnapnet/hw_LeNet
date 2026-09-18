"""Float MNIST inference and accuracy evaluation for the canonical LeNet."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5


def project_root():
    return Path(__file__).resolve().parents[1]


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            correct += (logits.argmax(1).cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total, correct, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="weights/lenet_float.pt")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", default="results/baseline_accuracy.json")
    args = parser.parse_args()
    root = project_root()
    checkpoint = root / args.checkpoint
    output = root / args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = datasets.MNIST(root=str(root / "data"), train=False, download=False, transform=transforms.ToTensor())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = LeNet5().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state))
    accuracy, correct, total = evaluate(model, loader, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"accuracy": accuracy, "correct": correct, "total": total, "device": str(device)}, indent=2), encoding="utf-8")
    (output.parent / "baseline_accuracy.txt").write_text(
        "MNIST test accuracy: {:.6f}\nCorrect: {} / {}\nDevice: {}\n".format(accuracy, correct, total, device),
        encoding="utf-8",
    )
    print("MNIST test accuracy: {:.4f} ({}/{})".format(accuracy, correct, total))
    print("Saved:", output)


if __name__ == "__main__":
    main()
