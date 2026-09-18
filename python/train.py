"""Train the canonical Float LeNet baseline and save a portable checkpoint."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import LeNet5, model_metadata


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def project_root():
    return Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--checkpoint", default="weights/lenet_float.pt")
    args = parser.parse_args()
    root = project_root()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(root / "data"), train=True, download=False, transform=tfm)
    test_ds = datasets.MNIST(root=str(root / "data"), train=False, download=False, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_accuracy = -1.0
    best_epoch = 0
    checkpoint = root / args.checkpoint
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.numel()
        model.eval()
        test_correct = test_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                test_correct += (model(images.to(device)).argmax(1).cpu() == labels).sum().item()
                test_total += labels.numel()
        row = {"epoch": epoch, "train_loss": running_loss / total, "train_accuracy": correct / total, "test_accuracy": test_correct / test_total}
        history.append(row)
        print("epoch {epoch}: loss={train_loss:.4f}, train={train_accuracy:.4f}, test={test_accuracy:.4f}".format(**row))
        if row["test_accuracy"] > best_accuracy:
            best_accuracy = row["test_accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "metadata": model_metadata(),
                    "history": history.copy(),
                    "seed": args.seed,
                    "best_epoch": best_epoch,
                    "best_accuracy": best_accuracy,
                },
                checkpoint,
            )
            print("  New best checkpoint: epoch {} (test={:.4f})".format(best_epoch, best_accuracy))
    metrics = root / "results" / "baseline_training.json"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps({"checkpoint": str(checkpoint), "device": str(device), "best_epoch": best_epoch, "best_accuracy": best_accuracy, "history": history}, indent=2), encoding="utf-8")
    print("Best checkpoint: {} (epoch {}, test={:.4f})".format(checkpoint, best_epoch, best_accuracy))
    print("Saved metrics:", metrics)


if __name__ == "__main__":
    main()
