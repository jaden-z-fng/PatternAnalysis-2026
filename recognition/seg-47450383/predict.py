import os
import time
 
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
from modules import UNet2D               
from dataset import HipMRIDataset

TEST_IMG_DIR = "keras_slices_test"
TEST_SEG_DIR = "keras_slices_seg_test"
WEIGHTS_PATH = "unet2d.pth"
OUT_DIR = "predictions"
NUM_CLASSES = 6                           
BATCH_SIZE = 8
NUM_FIGURES = 3                            # how many best / worst slices to plot
EPS = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_label_map(mask):
    """Accept a one-hot mask (B,C,H,W), a label map (B,H,W) or (B,1,H,W)."""
    if mask.dim() == 4 and mask.shape[1] == 1:
        return mask[:, 0].long()
    if mask.dim() == 4:
        return mask.argmax(dim=1)
    return mask.long()
 
 
def per_class_counts(pred, target, num_classes):
    """Return intersection, pred-area and target-area per class for one batch."""
    inter = torch.zeros(num_classes)
    p_area = torch.zeros(num_classes)
    t_area = torch.zeros(num_classes)
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter[c] = (p & t).sum().item()
        p_area[c] = p.sum().item()
        t_area[c] = t.sum().item()
    return inter, p_area, t_area
 
 
def plot_case(image, gt, pred, title, save_path):
    """Four panels: input, ground truth, prediction, difference overlay."""
    wrong = gt != pred
    fp = (pred > 0) & (gt == 0)                      # predicted structure, none in GT
    fn = (gt > 0) & (pred == 0)                      # missed structure
    diff = np.zeros(gt.shape + (3,))
    diff[fp] = [1, 0, 0]                             # red   = false positive
    diff[fn] = [0, 0, 1]                             # blue  = false negative
    diff[wrong & ~fp & ~fn] = [1, 1, 0]              # yellow = wrong class
 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(diff, alpha=0.6)
    axes[3].set_title("Difference (red FP, blue FN, yellow wrong class)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    # ---- data ----
    test_set = HipMRIDataset(TEST_IMG_DIR, TEST_SEG_DIR)   # ADAPT: constructor args
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
 
    # ---- model ----
    model = UNet2D(in_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
 
    # ---- inference ----
    inter = torch.zeros(NUM_CLASSES)
    p_area = torch.zeros(NUM_CLASSES)
    t_area = torch.zeros(NUM_CLASSES)
    slice_scores = []        # (slice index, mean Dice over classes present in GT)
    cache = {}               # slice index -> (image, gt, pred) kept for plotting
    n_slices = 0
    total_time = 0.0
 
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            target = to_label_map(masks).to(device)
 
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - start
 
            pred = logits.argmax(dim=1)
 
            i, p, t = per_class_counts(pred.cpu(), target.cpu(), NUM_CLASSES)
            inter += i
            p_area += p
            t_area += t
 
            # per-slice Dice so we can find the best and worst cases
            for b in range(images.shape[0]):
                dices = []
                for c in range(1, NUM_CLASSES):      # skip background
                    gt_c = target[b] == c
                    if gt_c.sum() == 0:
                        continue
                    pr_c = pred[b] == c
                    d = (2 * (gt_c & pr_c).sum() + EPS) / (gt_c.sum() + pr_c.sum() + EPS)
                    dices.append(d.item())
                if dices:
                    slice_scores.append((n_slices, float(np.mean(dices))))
                cache[n_slices] = (
                    images[b, 0].cpu().numpy(),
                    target[b].cpu().numpy(),
                    pred[b].cpu().numpy(),
                )
                n_slices += 1
 
    # ---- metrics ----
    dice = (2 * inter + EPS) / (p_area + t_area + EPS)
    iou = (inter + EPS) / (p_area + t_area - inter + EPS)
 
    print(f"Test slices: {n_slices}")
    print(f"{'Class':<8}{'Dice':>8}{'IoU':>8}")
    for c in range(NUM_CLASSES):
        print(f"{c:<8}{dice[c].item():>8.4f}{iou[c].item():>8.4f}")
    print(f"Mean Dice (excl. background): {dice[1:].mean().item():.4f}")
    print(f"Mean IoU  (excl. background): {iou[1:].mean().item():.4f}")
    print(f"Inference time: {1000 * total_time / n_slices:.2f} ms per slice")
    if device.type == "cuda":
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024 ** 2:.0f} MB")
 
    # ---- visualisations: worst and best slices ----
    slice_scores.sort(key=lambda s: s[1])
    worst = slice_scores[:NUM_FIGURES]
    best = slice_scores[-NUM_FIGURES:]
    for tag, chosen in (("worst", worst), ("best", best)):
        for idx, score in chosen:
            image, gt, pred = cache[idx]
            plot_case(image, gt, pred,
                      f"{tag} case: slice {idx}, mean Dice {score:.3f}",
                      os.path.join(OUT_DIR, f"{tag}_slice_{idx}.png"))
    print(f"Saved figures to {OUT_DIR}/")


