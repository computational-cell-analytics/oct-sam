from typing import List, Optional
import warnings

import numpy as np

# From top to bottom
LAYERS = {
    "RFNL": "Retinal Nerve Fiber Layer",
    "GCIPL": "Ganglion Cell/Inner Plexiform Layer",
    "INL": "Inner Nuclear Layer",
    "OPL": "Outer Plexiform Layer",
    "ONL": "Outer Nuclear Layer",
    "EZ": "Ellipsoid Zone",
    "RPE": "Retinal Pigment Epithelium",
}


def find_layer_order(seg: np.ndarray) -> Optional[List[int]]:
    """Identify the order of segmentation layers.

    The function checks every column of the image if it contains all layers.
    If such a column is found, the order (from top to bottom) of the segmentation IDs is returned.

    Args:
        seg: Segmentation mask.

    Returns:
        Ordered list of segmentation IDs from top to bottom.
    """
    unique_ids = np.unique(seg)[1:]
    height, width = seg.shape
    for x in range(width):
        col = seg[:, x]
        col = list(dict.fromkeys(list(col)))
        col_ids = [int(c) for c in col if c != 0]
        if all([i in col_ids for i in unique_ids]):
            return col_ids
    return None


def identify_layers(seg: np.ndarray, expected_number_of_layers: Optional[int] = None) -> Optional[dict]:
    """Match the layer identification to the segmentation.

    If all the segmentation IDs are in the same column, they are ordered from top to bottom.
    Then, they are matched to labels from the label dictionary.
    Once the segmentation of thin layers has improved and the necessity for such a process can be determined,
    a more sophisticated identification can be implemented.

    Args:
        seg: The OCT segmentation.
        expected_number_of_layers: Expected number of layers in the segmentation.

    Returns:
        The label dictionary matching layer identifiers (keys) to segmentation IDs (values).
    """
    col_ids = find_layer_order(seg)
    if col_ids is None:
        warnings.warn("Could not determine the order of labels.")
        return None

    n_ids = len(col_ids)
    if expected_number_of_layers is not None and n_ids != expected_number_of_layers:
        warnings.warn(
            f"The number of expected layers {expected_number_of_layers} does not match the actual number {n_ids}."
        )
        return None

    # The layers degrade from the bottom, so if layers are missing we can just index the first n-ids.
    # layer_names = list(LAYERS.keys())[:n_ids]
    # The automatic assignment of the correct layer names is currently too inaccurate.
    layer_names = [f"layer_{str(i+1).zfill(2)}" for i in range(n_ids)]
    seg_label_dict = {col_id: name for col_id, name in zip(col_ids, layer_names)}
    return seg_label_dict
