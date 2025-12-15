from typing import List, Optional

import numpy as np

LABEL_DICT = {
    4: ["RFNL", "GCIPL", "INL", "OPL", "RPE"],
    6: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "RPE"],
    7: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "EZ", "RPE"],
}

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
    """Identify order of segmentation layers.
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


def identify_layers(
    seg: np.ndarray,
    label_dict: dict = LABEL_DICT,
) -> Optional[dict]:
    """Match the layer identification to the segmentation.
    If all the segmentation IDs are in the same column, they are ordered from top to bottom.
    Then, they are matched to labels from the label dictionary.
    Once the segmentation of thin layers has improved and the necessity for such a process can be determined,
    a more sophisticated identification can be implemented.

    Args:
        seg: Segmentation.

    Returns:
        Label dictionary matching layer identifiers (keys) to segmentation IDs (values).
    """
    col_ids = find_layer_order(seg)
    seg_label_dict = {}
    if len(col_ids) in list(label_dict.keys()):
        label_list = label_dict[len(col_ids)]
        for (lid, cid) in zip(label_list, col_ids):
            seg_label_dict[lid] = cid
        return seg_label_dict
    else:
        if col_ids:
            print("Not all segmentation IDs are present in a single column.")
        else:
            print(f"Could not identify cross-section of {len(col_ids)} layers. Check label dictionary.")
        return None
