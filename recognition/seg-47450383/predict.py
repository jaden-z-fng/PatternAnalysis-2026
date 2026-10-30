import torch

from modules import UNet2D           # your model
from dataset import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
