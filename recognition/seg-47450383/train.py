import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from modules import UNet2D
from dataset import get_dataloaders
import torch.nn.functional as F


NUM_CLASSES = 6
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 16
SAVE_PATH = "unet2d.pth"
# DATA_ROOT = "."
DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_one_hot(masks):
    """[B, H, W] integer labels -> [B, C, H, W] one-hot floats."""
    return F.one_hot(masks.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def update_counts(logits, target, counts):
    """Accumulate per-class intersection / pred / target pixel counts."""
    pred = F.one_hot(
        logits.argmax(dim=1), NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    counts["inter"] += (pred * target).sum(dims)
    counts["pred"] += pred.sum(dims)
    counts["target"] += target.sum(dims)


def dice_from_counts(counts, eps=1e-6):
    inter, p, t = counts["inter"], counts["pred"], counts["target"]
    return (2 * inter + eps) / (p + t + eps)


def new_counts():
    z = lambda: torch.zeros(NUM_CLASSES, device=DEVICE)
    return {"inter": z(), "pred": z(), "target": z()}


class DiceCELoss(torch.nn.Module):
    def __init__(self, ce_weight=0.5, eps=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice_per_class = (2 * inter + self.eps) / (denom + self.eps)
        dice_loss = 1 - dice_per_class[1:].mean()

        ce_loss = torch.nn.functional.cross_entropy(logits, target)
        return (1 - self.ce_weight) * dice_loss + self.ce_weight * ce_loss


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for images, masks in loader:
        images, masks = images.to(DEVICE), to_one_hot(masks.to(DEVICE))
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * images.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, counts = 0.0, new_counts()
    for images, masks in loader:
        images, masks = images.to(DEVICE), to_one_hot(masks.to(DEVICE))
        logits = model(images)
        total += criterion(logits, masks).item() * images.size(0)
        update_counts(logits, masks, counts)
    return total / len(loader.dataset), dice_from_counts(counts)


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="validation")
    plt.title("Training and validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()


def main():
    set_seed(SEED)
    train_loader, val_loader, test_loader = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)

    model = UNet2D(n_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total, trainable = count_params(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": []}
    best_dice = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        mean_dice = val_dice[1:].mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Dice {mean_dice:.4f}")

        if mean_dice > best_dice:  # keep the best model on validation
            best_dice = mean_dice
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training time: {(time.time() - start) / 60:.1f} min")
    if DEVICE.type == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    plot_history(history)

    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    _, test_dice = evaluate(model, test_loader, criterion)
    print("\nTest results (per class):")
    for c in range(NUM_CLASSES):
        print(f"  class {c}: Dice {test_dice[c]:.4f}")
    print(f"  mean (foreground): Dice {test_dice[1:].mean():.4f}")


if __name__ == "__main__":
    main()