import colorsys
from typing import List, Optional

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


def _hex_to_rgba(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([r / 255, g / 255, b / 255, 1.0])


def _generate_warning_colors(n: int = 20, seed: int = 42) -> list:
    """Generate n visually distinct warning colors using golden-ratio hue spacing.

    Consecutive hues jump by ~222° so the sequence looks random rather than
    a smooth gradient.  Saturation and value are fixed at 1.0, making every
    color fully vivid and immediately distinguishable from the more muted
    layer palettes regardless of hue overlap.
    """
    rng = np.random.default_rng(seed)
    golden_ratio_conjugate = 0.6180339887498949
    h = float(rng.random())
    colors = []
    for _ in range(n):
        h = (h + golden_ratio_conjugate) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        colors.append(np.array([r, g, b, 1.0]))
    return colors


_WARNING_COLORS_BRIGHT = _generate_warning_colors(20)

# Fixed colors for the 7 retinal layers (label IDs 1–7), inner → outer retina.
LAYER_COLORS = {
    None: np.zeros(4),                   # transparent
    0:    np.zeros(4),                   # background
    1:    _hex_to_rgba("#782506"),        # RNFL
    2:    _hex_to_rgba("#5bd5f8"),        # GCIPL
    3:    _hex_to_rgba("#9289e8"),        # INL
    4:    _hex_to_rgba("#6c02c1"),        # OPL
    5:    _hex_to_rgba("#473a9f"),        # ONL
    6:    _hex_to_rgba("#abec8a"),        # EZ
    7:    _hex_to_rgba("#8fadb2"),        # RPE
}

LAYER_COLORS_CUSTOM = {
    None: np.zeros(4),                   # transparent
    0:    np.zeros(4),                   # background
    1:    _hex_to_rgba("#218561"),        # RNFL
    2:    _hex_to_rgba("#833733"),        # GCIPL
    3:    _hex_to_rgba("#73aebd"),        # INL
    4:    _hex_to_rgba("#d58940"),        # OPL
    5:    _hex_to_rgba("#972892"),        # ONL
    6:    _hex_to_rgba("#d7ca52"),        # EZ
    7:    _hex_to_rgba("#1b41a4"),        # RPE
}


def get_layer_colormap(style: str = "default"):
    """Return a colormap for the 7 retinal layer label IDs.

    Args:
        style: Color style to apply.

            * ``"default"`` — fixed project colors for IDs 1–7, bright warning
              colors for IDs 8–255.
            * ``"custom"`` — alternative palette for IDs 1–7, same warning
              colors for IDs 8–255.
            * ``"check"`` — all IDs 1–7 green, all other IDs red; useful for
              a quick visual validation that every pixel has been assigned a
              valid layer label.
            * ``"random"`` — leaves coloring to napari; returns ``None`` so
              callers skip setting the colormap.

    Returns:
        A ``DirectLabelColormap`` instance, or ``None`` for ``"random"``.
    """
    if style == "random":
        return None
    from napari.utils.colormaps import direct_colormap
    if style == "check":
        green = np.array([0.0, 1.0, 0.0, 1.0])
        red = np.array([1.0, 0.0, 0.0, 1.0])
        colors: dict = {None: np.zeros(4), 0: np.zeros(4)}
        for i in range(1, 8):
            colors[i] = green
        for i in range(8, 256):
            colors[i] = red
        return direct_colormap(colors)
    colors = dict(LAYER_COLORS if style == "default" else LAYER_COLORS_CUSTOM)
    for i in range(8, 256):
        colors[i] = _WARNING_COLORS_BRIGHT[i % len(_WARNING_COLORS_BRIGHT)]
    return direct_colormap(colors)


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
