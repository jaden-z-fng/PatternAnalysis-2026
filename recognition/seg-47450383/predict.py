import torch

from modules import UNet2D           # your model
from dataset import get_dataloaders


NUM_CLASSES = 6 
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
