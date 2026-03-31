"""
Visual Recognition using Deep Learning - HW1
ResNet Image Classification (100 classes)

Usage:
    # Training (baseline)
    python main.py --mode train --data_root ./data

    # Training with CutMix
    python main.py --mode train --data_root ./data --cutmix

    # Inference
    python main.py --mode inference --data_root ./data --ckpt best_model_cutmix.pth
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import glob


# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH_SIZE = 64
NUM_CLASSES = 100
NUM_EPOCHS = 80
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
CUTMIX_PROB = 0.5
CUTMIX_ALPHA = 1.0


# ── DataLoader & Augmentation ─────────────────────────────────────────────────


def get_transforms(train: bool) -> transforms.Compose:
    """Return augmentation pipeline for train or val/test.

    Args:
        train: if True, return training augmentation; otherwise val/test.

    Returns:
        A torchvision Compose transform pipeline.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )


# ── SortedImageFolder─────────────────────────────────────────────────────────
class SortedImageFolder(ImageFolder):

    def find_classes(self, directory):
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        classes.sort(key=lambda x: int(x))
        class_to_idx = {cls: int(cls) for cls in classes}
        return classes, class_to_idx


def get_dataloaders(data_root: str, batch_size: int):
    train_ds = SortedImageFolder(
        root=os.path.join(data_root, "train"),
        transform=get_transforms(train=True),
    )
    val_ds = SortedImageFolder(
        root=os.path.join(data_root, "val"),
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


# ── Model ─────────────────────────────────────────────────────────────────────


def build_model(num_classes: int = 100, freeze_backbone: bool = False) -> nn.Module:
    """Load pretrained ResNet-101 and replace the final FC layer.

    Args:
        num_classes: number of output categories.
        freeze_backbone: if True, freeze all layers except the FC head.

    Returns:
        Modified ResNet-101 model.
    """
    model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ── Training Loop (Baseline) ──────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch without CutMix.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


# ── CutMix ────────────────────────────────────────────────────────────────────


def cutmix_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float = 1.0):
    """Apply CutMix augmentation to a mini-batch.

    Args:
        images: input batch of shape (B, C, H, W).
        labels: ground truth labels of shape (B,).
        alpha: Beta distribution parameter.

    Returns:
        Tuple of (mixed_images, labels_a, labels_b, lam).
    """
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(batch_size)
    labels_a = labels
    labels_b = labels[rand_index]

    h, w = images.size(2), images.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[rand_index, :, y1:y2, x1:x2]

    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed_images, labels_a, labels_b, lam


def cutmix_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Compute mixed loss: lam * L(a) + (1 - lam) * L(b)."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def train_one_epoch_cutmix(
    model, loader, optimizer, criterion, device, cutmix_prob: float = 0.5
):
    """Run one training epoch with CutMix applied at probability cutmix_prob.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if np.random.rand() < cutmix_prob:
            images, labels_a, labels_b, lam = cutmix_batch(
                images, labels, alpha=CUTMIX_ALPHA
            )
            outputs = model(images)
            loss = cutmix_criterion(criterion, outputs, labels_a, labels_b, lam)
            preds = outputs.argmax(dim=1)
            correct += (
                lam * (preds == labels_a).sum().item()
                + (1 - lam) * (preds == labels_b).sum().item()
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            correct += (outputs.argmax(dim=1) == labels).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        total += images.size(0)

    return total_loss / total, correct / total


# ── Training ──────────────────────────────────────────────────────────────────


def train(args):
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = get_dataloaders(args.data_root, BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_model(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {total_params:.2f}M (limit: 100M)")

    optimizer = SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    tag = "cutmix" if args.cutmix else "baseline"
    ckpt_path = f"best_model_{tag}.pth"
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        if args.cutmix:
            train_loss, train_acc = train_one_epoch_cutmix(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                cutmix_prob=CUTMIX_PROB,
            )
        else:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            saved = " <- best"

        print(
            f"[{tag}] Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} Acc {val_acc:.4f}{saved}"
        )

    print(f"\nBest val accuracy: {best_val_acc:.4f} | Checkpoint: {ckpt_path}")
    plot_curves(history, tag)


def plot_curves(history: dict, tag: str):
    """Save training curves as PNG for the report."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Acc")
    ax2.plot(epochs, history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"training_curve_{tag}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")


# ── Inference ─────────────────────────────────────────────────────────────────


class TestDataset(Dataset):
    """Load test images from a flat folder with no class subfolders."""

    def __init__(self, test_dir: str, transform):
        self.paths = sorted(
            glob.glob(os.path.join(test_dir, "*.jpg"))
            + glob.glob(os.path.join(test_dir, "*.jpeg"))
            + glob.glob(os.path.join(test_dir, "*.png")),
            key=lambda p: os.path.basename(p),
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {test_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), os.path.basename(self.paths[idx])


def inference(args):
    """Run inference on flat test folder and output prediction.csv."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    state_dict = ckpt

    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    test_dir = os.path.join(args.data_root, "test")
    test_ds = TestDataset(test_dir, transform=get_transforms(train=False))
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    all_names, all_preds = [], []
    with torch.no_grad():
        for images, names in test_loader:
            outputs = model(images.to(device))
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_names.extend(names)

    with open("prediction.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "pred_label"])
        for name, pred in zip(all_names, all_preds):
            name_wo_ext = os.path.splitext(name)[0]
            writer.writerow([name_wo_ext, pred])

    print(f"Saved {len(all_preds)} predictions -> prediction.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="HW1 ResNet-101 Image Classification")
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--ckpt", type=str, default="best_model_baseline.pth")
    parser.add_argument("--cutmix", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        inference(args)
