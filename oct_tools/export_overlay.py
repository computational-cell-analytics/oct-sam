import os
from typing import List, Optional, Tuple

import numpy as np
import tifffile
from h5py import File
from imageio.v3 import imread, imwrite

from oct_tools.layer_information import _WARNING_COLORS_BRIGHT, LAYER_COLORS, LAYER_COLORS_CUSTOM

OVERLAY_ALPHA = 0.4


def _load_volume(path: str, keys: Tuple[str, ...], rgb_to_gray: bool = False) -> np.ndarray:
    if path.endswith(".h5"):
        with File(path, "r") as f:
            key = next((k for k in keys if k in f), None)
            if key is None:
                raise KeyError(f"H5 file {path} must contain one of the datasets {keys}.")
            data = f[key][:]
    else:
        data = imread(path)
    if data.ndim == 3 and data.shape[-1] == 3 and rgb_to_gray:
        data = data.mean(axis=-1).astype(data.dtype)
    if data.ndim == 2:
        data = data[np.newaxis]
    elif data.ndim != 3:
        raise ValueError(f"{path} must hold a 2D or a 3D array, got {data.ndim} dimensions.")
    return data


def _color_lookup(color_style: str) -> np.ndarray:
    """Build an RGB lookup table indexed by label ID.

    The palette dictionaries are read directly instead of going through get_layer_colormap, which
    imports napari. Indexing by ID rather than by rank keeps the colors stable when a layer is
    missing from a scan.
    """
    colors = LAYER_COLORS if color_style == "default" else LAYER_COLORS_CUSTOM
    lut = np.zeros((256, 3), dtype=np.float32)
    for label_id, color in colors.items():
        if isinstance(label_id, int) and 0 < label_id < 256:
            lut[label_id] = color[:3]
    # IDs 8-255 are not valid layers. Give them the same vivid warning colors as napari, so that
    # a segmentation which post-processing did not reduce to 7 layers stands out.
    for label_id in range(8, 256):
        lut[label_id] = _WARNING_COLORS_BRIGHT[label_id % len(_WARNING_COLORS_BRIGHT)][:3]
    return lut


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    lower, upper = image.min(), image.max()
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.round(255 * (image - lower) / (upper - lower)).astype(np.uint8)


def _blend(image: np.ndarray, segmentation: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Blend the layer colors onto the grayscale B-scans."""
    gray = _to_uint8(image).astype(np.float32)[..., np.newaxis] / 255
    weight = OVERLAY_ALPHA * (segmentation > 0).astype(np.float32)[..., np.newaxis]
    rgb = gray * (1 - weight) + lut[segmentation] * weight
    return np.round(255 * rgb).astype(np.uint8)


def export_overlay(
    image_path: str,
    segmentation_path: str,
    output_path: str,
    mode: str = "composite",
    color_style: str = "custom",
    slices: Optional[List[int]] = None,
):
    """Export B-scans and their annotation as a single file, for review outside napari.

    Args:
        image_path: Path to the image (TIF or H5).
        segmentation_path: Path to the segmentation (TIF or H5).
        output_path: Output file. A TIF holds a whole stack, a PNG only a single B-scan.
        mode: Either "composite" for a two-channel ImageJ TIF, in which the annotation can be
            switched off, or "rgb" for a flat overlay that any image viewer can show.
        color_style: Color palette for the retinal layers, "default" or "custom".
        slices: B-scans to export. All of them by default.
    """
    if mode not in ("composite", "rgb"):
        raise ValueError(f"Invalid mode {mode}. Choose either 'composite' or 'rgb'.")
    if mode == "composite" and not output_path.endswith((".tif", ".tiff")):
        raise ValueError(f"Mode 'composite' writes a TIF, but the output path is {output_path}.")

    image = _load_volume(image_path, ("image",), rgb_to_gray=True)
    segmentation = _load_volume(segmentation_path, ("segmentation", "seg"))
    if image.shape != segmentation.shape:
        raise ValueError(f"Image shape {image.shape} does not match segmentation shape {segmentation.shape}.")
    if segmentation.max() > 255:
        raise ValueError(f"Label ID {segmentation.max()} exceeds 255 and cannot be exported. Post-process first.")

    if slices is not None:
        image, segmentation = image[slices], segmentation[slices]

    output_folder = os.path.dirname(output_path)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    if mode == "composite":
        # ImageJ reads (Z, C, Y, X) and shows the two channels as one composite image.
        stack = np.stack([_to_uint8(image), segmentation.astype(np.uint8)], axis=1)
        tifffile.imwrite(output_path, stack, imagej=True, metadata={"axes": "ZCYX", "mode": "composite"})
    else:
        rgb = _blend(image, segmentation, _color_lookup(color_style))
        if output_path.endswith(".png"):
            if len(rgb) != 1:
                raise ValueError(f"PNG holds a single B-scan, but {len(rgb)} were selected. Use a TIF.")
            imwrite(output_path, rgb[0])
        else:
            imwrite(output_path, rgb)

    print(f"Exported {len(image)} B-scan(s) to {output_path}.")
