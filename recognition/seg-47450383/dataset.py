import os
import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_nifti_2d(path, norm_image=False, dtype=np.float32):
    """Load one 2D slice from a .nii.gz file (same handling as load_data_2D)."""
    img = nib.load(path).get_fdata(caching="unchanged")

    if img.ndim == 3:
        img = img[:, :, 0]  # remove extra dimension in HipMRI slices

    img = img.astype(dtype)

    if norm_image:
        img = (img - img.mean()) / (img.std() + 1e-8)

    return img


class HipMRIDataset(Dataset):
    """2D HipMRI slice dataset.

    Expects folders like:
        root/keras_slices_train        (images)
        root/keras_slices_seg_train    (masks)
    and the same for test / validate.

    Image files look like  case_040_week_0_slice_0.nii.gz
    Mask files look like   seg_040_week_0_slice_0.nii.gz

    Returns:
        image: float tensor (1, H, W), normalised to zero mean / unit variance
        mask:  long tensor (H, W) of class indices
               (or float one-hot (num_classes, H, W) if one_hot=True)
    """

    def __init__(self, root, split="train", num_classes=6, norm_image=True, early_stop=False):
        self.num_classes = num_classes

        image_dir = os.path.join(root, f"keras_slices_{split}")
        mask_dir = os.path.join(root, f"keras_slices_seg_{split}")

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.nii.gz")))
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No .nii.gz files found in {image_dir}")

        # Match each image to its mask by swapping the "case" prefix for "seg"
        mask_paths = []
        for p in image_paths:
            name = os.path.basename(p)
            mask_name = "seg" + name[len("case"):] if name.startswith("case") else name
            mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Missing mask for {name}: {mask_path}")
            mask_paths.append(mask_path)

        if early_stop:  # quick testing with a small subset
            image_paths, mask_paths = image_paths[:21], mask_paths[:21]

        # Pre-load everything into memory (slices are small), like load_data_2D
        first = load_nifti_2d(image_paths[0])
        rows, cols = first.shape
        n = len(image_paths)

        self.images = np.zeros((n, rows, cols), dtype=np.float32)
        self.masks = np.zeros((n, rows, cols), dtype=np.int64)

        for i in tqdm(range(n), desc=f"Loading {split}"):
            self.images[i] = load_nifti_2d(image_paths[i], norm_image=norm_image)
            # Masks are label maps: no normalisation, round to integer classes
            self.masks[i] = np.rint(load_nifti_2d(mask_paths[i])).astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(self.masks[idx])                 # (H, W)

        return image, mask


def get_dataloaders(root, batch_size=16, num_classes=6, num_workers=0):
    """Build train / validation / test dataloaders."""
    train_set = HipMRIDataset(root, "train", num_classes)
    val_set = HipMRIDataset(root, "validate", num_classes)
    test_set = HipMRIDataset(root, "test", num_classes)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick check: python dataset.py /path/to/data
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "."
    ds = HipMRIDataset(root, "train", early_stop=True)
    img, msk = ds[0]
    print(len(ds), img.shape, img.dtype, msk.shape, msk.dtype, msk.unique())