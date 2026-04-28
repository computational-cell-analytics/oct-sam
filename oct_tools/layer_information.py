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

# Number of layers matched to layer order
LAYER_NUMBER_DICT = {
    3: ["RFNL", "GCIPL", "RPE"],
    4: ["RFNL", "GCIPL", "INL", "RPE"],
    5: ["RFNL", "GCIPL", "INL", "OPL", "RPE"],
    6: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "RPE"],
    7: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "EZ", "RPE"],
}

# Matching of layer and label ID
LAYER_LABEL_DICT = {
    "RFNL": 1,
    "GCIPL": 2,
    "INL": 3,
    "OPL": 4,
    "ONL": 5,
    "EZ": 6,
    "RPE": 7,
}

LAYER_MAPPING = {
    1: "RNFL",
    2: "GCIPL",
    3: "INL",
    4: "OPL",
    5: "ONL",
    6: "EZ",
    7: "RPE"
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


def identify_layers_naively(
    seg: np.ndarray,
    generic_names: bool = True,
) -> Optional[dict]:
    """Match the layer identification to the segmentation.

    If all the segmentation IDs are in the same column, they are ordered from top to bottom.
    Then, they are matched to labels from the label dictionary.

    Args:
        seg: The OCT segmentation.

    Returns:
        The label dictionary matching layer identifiers (keys) to segmentation IDs (values).
    """
    layer_order = find_layer_order(seg)
    if layer_order is None:
        print("No layer order could be identified.")
        return None
    else:
        number_layers = len(layer_order)
        if generic_names:
            layer_names = [f"layer_{str(i + 1).zfill(2)}" for i in range(number_layers)]
        elif number_layers in LAYER_NUMBER_DICT.keys():
            layer_names = LAYER_NUMBER_DICT[number_layers]
        else:
            print(f"No predefined set for {number_layers} layers.")
            return None
    seg_label_dict = {col_id: name for col_id, name in zip(layer_order, layer_names)}
    return seg_label_dict
