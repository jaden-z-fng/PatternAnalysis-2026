import nibabel as nib
import numpy as np
from tqdm import tqdm


def to_channels(arr: np.ndarray, dtype=np.uint8) -> np.ndarray:
    """Converts a multi-class label array into a one-hot encoded channel array."""
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c : c + 1][arr == c] = 1

    return res


# Load medical image functions
def load_data_2D(
    imageNames,
    normImage=False,
    categorical=False,
    dtype=np.float32,
    getAffines=False,
    early_stop=False,
):
    """Load medical image data from names/cases list into a pre-allocated array.

    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.

    Parameters:
    - normImage: bool (normalise the image to zero mean, unit variance)
    - early_stop: Stop loading prematurely (leaves array partially empty) for quick testing.
    """
    affines = []

    # Pre-allocate array based on the first image's shape
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching="unchanged")

    if len(first_case.shape) == 3:
        first_case = first_case[:, :, 0]  # Remove extra dimension if present

    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching="unchanged")  # Read disk only
        affine = niftiImage.affine

        if len(inImage.shape) == 3:
            inImage = inImage[:, :, 0]  # Extra dims handling for HipMRI study data

        inImage = inImage.astype(dtype)

        if normImage:
            inImage = (inImage - inImage.mean()) / (inImage.std() + 1e-8)

        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i, :, :, :] = inImage
        else:
            images[i, :, :] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images