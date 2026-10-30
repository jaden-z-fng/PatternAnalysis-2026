import os
import glob

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def load_nifti_2d(path, dtype=np.float32):
    """Load one 2D slice from a .nii.gz file (same handling as load_data_2D)."""
    img = nib.load(path).get_fdata(caching="unchanged")

    if img.ndim == 3:
        img = img[:, :, 0]  # remove extra dimension in HipMRI slices

    img = img.astype(dtype)

    return img

