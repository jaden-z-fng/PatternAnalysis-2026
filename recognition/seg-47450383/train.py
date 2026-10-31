import torch
import time
import matplotlib.pyplot as plt

from modules import UNet2D           # your model
from dataset import get_dataloaders


NUM_CLASSES = 6
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 16
SAVE_PATH = "unet2d.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_one_hot(masks):
    """[B, H, W] integer labels -> [B, C, H, W] one-hot floats."""
    return torch.nn.functional.one_hot(masks.long(), NUM_CLASSES).permute(0, 3, 1, 2).float()

@torch.no_grad()
def update_counts(logits, target, counts):
    """Accumulate per-class intersection / pred / target pixel counts."""
    pred = torch.nn.functional.one_hot(
        logits.argmax(dim=1), NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    counts["inter"] += (pred * target).sum(dims)
    counts["pred"] += pred.sum(dims)
    counts["target"] += target.sum(dims)


def dice_iou_from_counts(counts, eps=1e-6):
    inter, p, t = counts["inter"], counts["pred"], counts["target"]
    dice = (2 * inter + eps) / (p + t + eps)
    iou = (inter + eps) / (p + t - inter + eps)
    return dice, iou


def new_counts():
    z = lambda: torch.zeros(NUM_CLASSES, device=DEVICE)
    return {"inter": z(), "pred": z(), "target": z()}


class DiceCELoss(torch.nn.Module):
    def __init__(self, ce_weight=0.5, eps=1e-6):
        super().__init__()
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(self, logits, target):
        # Dice part
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice_loss = 1 - ((2 * inter + self.eps) / (denom + self.eps)).mean()

        # Cross-entropy part (accepts one-hot float targets)
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
    dice, iou = dice_iou_from_counts(counts)
    return total / len(loader.dataset), dice, iou

def plot_history(history):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="val")
    ax[0].set_title("Dice loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(epochs, history["val_dice"], label="val mean Dice")
    ax[1].plot(epochs, history["val_iou"], label="val mean IoU")
    ax[1].set_title("Validation metrics"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.close()

def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = UNet2D(nn_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}
    best_dice = 0.0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_dice, val_iou = evaluate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice.mean().item())
        history["val_iou"].append(val_iou.mean().item())

        print(f"Epoch {epoch:02d}/{EPOCHS} | train loss {train_loss:.4f} | "
              f"val loss {val_loss:.4f} | val Dice {val_dice.mean():.4f} | "
              f"val IoU {val_iou.mean():.4f}")

        if val_dice.mean().item() > best_dice:  # keep the best model on validation
            best_dice = val_dice.mean().item()
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Training time: {(time.time() - start) / 60:.1f} min")
    if DEVICE.type == "cuda":
        print(f"Peak GPU VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")

    plot_history(history)

