import torch
import time

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

class DiceLoss(torch.nn.Module):
    """Soft Dice loss only (no cross-entropy). Expects logits [B,C,H,W]
    and one-hot targets [B,C,H,W]."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)  # sum over batch + spatial, keep classes separate
        inter = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)  # per class
        return 1 - dice.mean()


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
    for images, masks in loader:
        images, masks = images.to(DEVICE), to_one_hot(masks.to(DEVICE))
        logits = model(images)
        total += criterion(logits, masks).item() * images.size(0)
    return total / len(loader.dataset),

def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = DiceLoss()
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

