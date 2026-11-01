from linecache import cache
import os
import time
import re
from collections import defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from torch.utils.data import DataLoader
 
from modules import UNet2D               
from dataset import HipMRIDataset

WEIGHTS_PATH = "unet2d.pth"
OUT_DIR = "predictions"
DATA_ROOT = "/home/jaden/COMP3710/keras_slices_data"
# DATA_ROOT = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
NUM_CLASSES = 6                           
BATCH_SIZE = 8
NUM_FIGURES = 3                            
EPS = 1e-6
NAME_RE = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+)")
EDGE_SLICES = 2
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

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

@torch.no_grad()
def predict_slice(model, dataset, idx):
    """Re-run the model on one test slice and return numpy arrays for plotting."""
    image, mask = dataset[idx]                      # (1,H,W), (H,W); test set is unaugmented
    logits = model(image.unsqueeze(0).to(device))
    pred = logits.argmax(dim=1)[0].cpu().numpy()
    return image[0].numpy(), mask.numpy(), pred
 
 
# def plot_case(image, gt, pred, title, save_path):
#     """Four panels: input, ground truth, prediction, difference overlay."""
#     wrong = gt != pred
#     fp = (pred > 0) & (gt == 0)                      # predicted structure, none in GT
#     fn = (gt > 0) & (pred == 0)                      # missed structure
#     diff = np.zeros(gt.shape + (3,))
#     diff[fp] = [1, 0, 0]                             # red   = false positive
#     diff[fn] = [0, 0, 1]                             # blue  = false negative
#     diff[wrong & ~fp & ~fn] = [1, 1, 0]              # yellow = wrong class
 
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
#     axes[0].imshow(image, cmap="gray")
#     axes[0].set_title("Input")
#     axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
#     axes[1].set_title("Ground truth")
#     axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
#     axes[2].set_title("Prediction")
#     axes[3].imshow(image, cmap="gray")
#     axes[3].imshow(diff, alpha=0.6)
#     axes[3].set_title("Difference (red FP, blue FN, yellow wrong class)")
#     for ax in axes:
#         ax.axis("off")
#     fig.suptitle(title)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=120)
#     plt.close(fig)

def plot_case(image, gt, pred, title, save_path):
    gt_p, pr_p = gt == 5, pred == 5
    fp = pr_p & ~gt_p                                # prostate predicted, not in GT
    fn = gt_p & ~pr_p                                # prostate in GT, missed

    diff = np.zeros(gt.shape + (4,))                 # RGBA, transparent by default
    diff[fp] = [1, 0, 0, 0.8]                        # red  = false positive
    diff[fn] = [0, 1, 1, 0.8]                        # cyan = false negative

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray");  axes[0].set_title("Input")
    axes[1].imshow(gt, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[1].set_title("Ground truth")
    axes[2].imshow(pred, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    axes[2].set_title("Prediction")
    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(diff, interpolation="nearest")
    axes[3].set_title("Prostate errors (red FP, cyan FN)")
    for ax in axes:
        ax.axis("off")
    cmap = plt.get_cmap("tab10")
    class_handles = [Patch(facecolor=cmap(c), label=f"{c}: {CLASS_NAMES[c]}")
                 for c in range(NUM_CLASSES)]
    axes[2].legend(handles=class_handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=7, frameon=False)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_name(name):
    case, week, sl = NAME_RE.search(name).groups()
    return f"{case}_w{week}", int(sl)

def add_position(records):
    """Find each case's prostate extent from the GT and locate every slice relative to it."""
    for r in records:
        r.update(pos=None, end_dist=None, end=None, inside=False)
    by_case = defaultdict(list)
    for r in records:
        by_case[r["case"]].append(r)
    for rs in by_case.values():
        present = [r["slice"] for r in rs if r["gt_area"] > 0]
        if not present:
            continue
        lo, hi = min(present), max(present)
        for r in rs:
            s = r["slice"]
            r["inside"] = lo <= s <= hi
            r["pos"] = (s - lo) / max(hi - lo, 1)
            r["end_dist"] = min(abs(s - lo), abs(s - hi))
            r["end"] = "low-index end" if abs(s - lo) <= abs(s - hi) else "high-index end"


def report_by_position(records, n_bins=5):
    inside = [r for r in records if r["inside"] and r["dice"] is not None]
    bins = defaultdict(list)
    for r in inside:
        bins[min(int(r["pos"] * n_bins), n_bins - 1)].append(r["dice"])

    print("\nProstate Dice along its extent (0 = first slice, 1 = last):")
    for k in range(n_bins):
        if bins[k]:
            print(f"  {k / n_bins:.1f}-{(k + 1) / n_bins:.1f}: "
                  f"{np.mean(bins[k]):.3f}  (n={len(bins[k])})")

    edge = [r["dice"] for r in inside if r["end_dist"] < EDGE_SLICES]
    mid = [r["dice"] for r in inside if r["end_dist"] >= EDGE_SLICES]
    if edge and mid:
        print(f"  Edge slices (<{EDGE_SLICES} from end): {np.mean(edge):.3f} (n={len(edge)})"
              f" | Central slices: {np.mean(mid):.3f} (n={len(mid)})")

    spurious = [r for r in records if r["gt_area"] == 0 and r["pr_area"] > 0]
    print(f"  Slices with no GT prostate but predicted prostate: {len(spurious)}")


def pick_failures(records, k):
    edge = sorted((r for r in records
                   if r["inside"] and r["end_dist"] < EDGE_SLICES and r["dice"] is not None),
                  key=lambda r: r["dice"])
    artifacts = sorted((r for r in records if r["gt_area"] == 0 and r["pr_area"] > 0),
                       key=lambda r: -r["pr_area"])
    centre = sorted((r for r in records
                     if r["inside"] and r["end_dist"] >= EDGE_SLICES and r["dice"] is not None),
                    key=lambda r: -r["dice"])
    return edge[:k], artifacts[:k], centre[:k]

def list_spurious(records):
    spurious = [r for r in records if r["gt_area"] == 0 and r["pr_area"] > 0]
    spurious.sort(key=lambda r: (r["case"], r["slice"]))

    print(f"\nSlices with no GT prostate but predicted prostate: {len(spurious)}")
    print(f"{'Slice file':<40}{'Pred px':>9}{'Dist to GT prostate':>22}")
    for r in spurious:
        dist = "no GT prostate in case" if r["end_dist"] is None else f"{r['end_dist']} slices"
        print(f"{r['name']:<40}{r['pr_area']:>9}{dist:>22}")
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    test_set = HipMRIDataset(DATA_ROOT, "test")
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
 
    # ---- model ----
    model = UNet2D(n_channels=1, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
 
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
 
    # ---- inference ----
    inter = torch.zeros(NUM_CLASSES)
    p_area = torch.zeros(NUM_CLASSES)
    t_area = torch.zeros(NUM_CLASSES)  
    records = []          
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
 
            for b in range(images.shape[0]):
                name = test_set.names[n_slices]
                case, sl = parse_name(name)
                gt_p = target[b] == 5
                pr_p = pred[b] == 5
                gt_area, pr_area = gt_p.sum().item(), pr_p.sum().item()
                pdice = None
                if gt_area > 0:
                    pdice = (2 * (gt_p & pr_p).sum().item() + EPS) / (gt_area + pr_area + EPS)
                records.append(dict(idx=n_slices, name=name, case=case, slice=sl,
                                    gt_area=gt_area, pr_area=pr_area, dice=pdice))

                
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
 
    add_position(records)
    report_by_position(records)
    list_spurious(records)

    edge, artifacts, centre = pick_failures(records, NUM_FIGURES)
    groups = (("edge_worst", edge), ("artifact", artifacts), ("central_best", centre))
    for tag, chosen in groups:
        for r in chosen:
            image, gt, pred = predict_slice(model, test_set, r["idx"])
            d = "n/a" if r["dice"] is None else f"{r['dice']:.3f}"
            title = (f"{tag}: {r['name']} | prostate Dice {d} | "
                    f"GT area {r['gt_area']}, pred area {r['pr_area']}")
            stem = r["name"].replace(".nii.gz", "")
            plot_case(image, gt, pred, title, os.path.join(OUT_DIR, f"{tag}_{stem}.png"))

    r = next(r for r in records if r["name"] == "case_040_week_4_slice_2.nii.gz")
    image, gt, pred = cache[r["idx"]]
    plot_case(image, gt, pred, "far artifact: " + r["name"],
          os.path.join(OUT_DIR, "far_artifact_week4_slice2.png"))
    print(f"Saved figures to {OUT_DIR}/")

if __name__ == "__main__":
       main()
