import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from modules import UNet2D, ImprovedUNet2D
from dataset import get_dataloaders
import torch.nn.functional as F


NUM_CLASSES = 6
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 16
SAVE_PATH = "unet2d.pth"
#SAVE_PATH = "improved_unet2d.pth"
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

        ce_loss = F.cross_entropy(logits, target)
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
    total = 0.0
    inter = torch.zeros(NUM_CLASSES, device=DEVICE)
    pred_sum = torch.zeros(NUM_CLASSES, device=DEVICE)
    target_sum = torch.zeros(NUM_CLASSES, device=DEVICE)

    for images, masks in loader:
        images, masks = images.to(DEVICE), to_one_hot(masks.to(DEVICE))
        logits = model(images)
        total += criterion(logits, masks).item() * images.size(0)

        pred = to_one_hot(logits.argmax(dim=1))
        inter += (pred * masks).sum((0, 2, 3))
        pred_sum += pred.sum((0, 2, 3))
        target_sum += masks.sum((0, 2, 3))

    dice = (2 * inter + 1e-6) / (pred_sum + target_sum + 1e-6)
    return total / len(loader.dataset), dice


def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].set_title("Training and validation loss")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("loss")
    ax[0].legend()

    ax[1].plot(epochs, history["val_dice"], label="validation Dice", color="tab:green")
    ax[1].set_title("Validation Dice score (mean foreground)")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("Dice")
    ax[1].set_ylim(0, 1)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig("training_curves.png", dpi=150)
    plt.close(fig)

def main():
    set_seed(SEED)
    train_loader, val_loader, test_loader = get_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE)

    model = UNet2D(n_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    # model = ImprovedUNet2D(n_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    total, trainable = count_params(model)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        mean_dice = val_dice[1:].mean().item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(mean_dice)

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Dice {mean_dice:.4f}")

        if mean_dice > best_dice:  # keep the best model on validation
            best_dice = mean_dice
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training time: {(time.time() - start) / 60:.1f} min")
    if DEVICE.type == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    plot_history(history)

if __name__ == "__main__":
    main()