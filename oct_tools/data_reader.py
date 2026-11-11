import os
from typing import Optional, Sequence

import h5py
import imageio.v3 as imageio
import numpy as np
import tifffile

from oct_tools.heyex_vol import read_heyex_vol

IMAGE_KEYS = ("image",)
# Inference output first, then the refined and the original manual annotation.
SEGMENTATION_KEYS = ("segmentation", "seg", "labels/edit_v3", "labels/original")


def _read_h5(path: str, keys: Sequence[str]) -> np.ndarray:
    with h5py.File(path, "r") as f:
        key = next((k for k in keys if isinstance(f.get(k), h5py.Dataset)), None)
        if key is None:
            raise KeyError(f"H5 file {path} must contain one of the datasets {tuple(keys)}.")
        if key != keys[0]:
            print(f"Reading dataset '{key}' from {path}.")
        return f[key][:]


def _read_tif(path: str) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        series = tif.series
        if len(series) == 1:
            return series[0].asarray()
        # A TIF written page by page can store each B-scan as its own series.
        shapes = sorted({s.shape for s in series})
        if len(shapes) > 1:
            raise ValueError(f"{path} holds {len(series)} images with different shapes {shapes}.")
        return np.stack([s.asarray() for s in series])


def read_volume(path: str, keys: Sequence[str] = IMAGE_KEYS) -> np.ndarray:
    """Read the B-scans of a TIF, H5 or Heyex VOL file.

    Args:
        path: TIF file (or any other format that imageio reads), H5 file or VOL file.
        keys: H5 datasets to try in this order. A nested key such as "labels/edit_v3" is valid.
            The function prints the dataset that it reads if it is not the first key.

    Returns:
        Array of shape (z, y, x). A file with a single B-scan gives z = 1. An RGB or RGBA B-scan becomes grayscale.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".h5":
        data = _read_h5(path, keys)
    elif extension == ".vol":
        data = read_heyex_vol(path)
    elif extension in (".tif", ".tiff"):
        data = _read_tif(path)
    else:
        data = imageio.imread(path)

    # A 3D array with 3 or 4 channels in the last axis is one RGB(A) B-scan, not a stack.
    if data.ndim == 3 and data.shape[-1] in (3, 4):
        data = data[..., :3].mean(axis=-1).astype(data.dtype)
    if data.ndim == 2:
        data = data[np.newaxis]
    if data.ndim != 3:
        raise ValueError(f"{path} must hold a 2D or a 3D array, got {data.ndim} dimensions.")
    return data


def read_bscan(path: str, z: Optional[int] = None, keys: Sequence[str] = IMAGE_KEYS) -> np.ndarray:
    """Read one B-scan of a TIF, H5 or Heyex VOL file.

    Args:
        path: TIF file (or any other format that imageio reads), H5 file or VOL file.
        z: Index of the B-scan in a stack. A file with a single B-scan ignores it.
        keys: H5 datasets to try in this order.

    Returns:
        Array of shape (y, x).
    """
    volume = read_volume(path, keys)
    if len(volume) == 1:
        return volume[0]
    if z is None:
        raise ValueError(f"{path} holds {len(volume)} B-scans. Select one of them.")
    if not 0 <= z < len(volume):
        raise ValueError(f"{path} holds {len(volume)} B-scans. B-scan {z} does not exist.")
    return volume[z]
