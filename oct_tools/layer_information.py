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


# Muddy/Brown warning colors cycled for unexpected label IDs (> 7).
_WARNING_COLORS_MUDDY = [
    _hex_to_rgba("#634849"),    # muddy red
    _hex_to_rgba("#574B63"),    # muddy purple
    _hex_to_rgba("#515C63"),    # muddy blue
    _hex_to_rgba("#506351"),    # muddy green
]

# Bright warning colors cycled for unexpected label IDs (> 7).
_WARNING_COLORS_BRIGHT = [
    np.array([1.0, 1.0, 0.0, 1.0]),  # yellow
    np.array([1.0, 0.0, 1.0, 1.0]),  # magenta
    np.array([1.0, 0.5, 0.0, 1.0]),  # orange
    np.array([0.0, 1.0, 0.0, 1.0]),  # lime
]

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

LAYER_COLORS_PASTEL = {
    None: np.zeros(4),                   # transparent
    0:    np.zeros(4),                   # background
    1:    _hex_to_rgba("#49B8A5"),        # RNFL
    2:    _hex_to_rgba("#B84953"),        # GCIPL
    3:    _hex_to_rgba("#4992B8"),        # INL
    4:    _hex_to_rgba("#B76649"),        # OPL
    5:    _hex_to_rgba("#AD49B8"),        # ONL
    6:    _hex_to_rgba("#B89E49"),        # EZ
    7:    _hex_to_rgba("#495AB8"),        # RPE
}


def get_layer_colormap(style: str = "default", warning_color_style: str = "bright"):
    """Return a colormap for the 7 retinal layer label IDs.

    Args:
        style: ``"default"`` uses the fixed project colors; ``"pastel"`` uses a
               softer palette; ``"random"`` leaves coloring to napari (returns
               ``None`` — callers should skip setting the colormap in that case).

    Returns:
        A ``DirectLabelColormap`` instance, or ``None`` for ``"random"``.

    Label IDs outside the expected range 0–7 receive distinct muddy warning
    colors. Entries for the full uint8 range (8–255) are pre-populated so that
    napari's precomputed array-rendering path also picks them up (it only
    considers keys that are explicitly present in the color dict).
    """
    if style == "random":
        return None
    from napari.utils.colormaps import direct_colormap
    colors = LAYER_COLORS if style == "default" else LAYER_COLORS_PASTEL
    if warning_color_style == "bright":
        warning_colors = _WARNING_COLORS_BRIGHT
    else:
            warning_colors = _WARNING_COLORS_MUDDY
    for i in range(8, 256):
        colors[i] = warning_colors[i % len(warning_colors)]
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
