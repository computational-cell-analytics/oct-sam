import numpy as np
from scipy.ndimage import binary_closing
from skimage.measure import label as label_binary

from oct_tools.layer_information import find_layer_order

LAYER_NUMBER_DICT = {
    3: ["RFNL", "GCIPL", "RPE"],
    4: ["RFNL", "GCIPL", "INL", "RPE"],
    5: ["RFNL", "GCIPL", "INL", "OPL", "RPE"],
    6: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "RPE"],
    7: ["RFNL", "GCIPL", "INL", "OPL", "ONL", "EZ", "RPE"],
}

LAYER_LABEL_DICT = {
    "RFNL": 1,
    "GCIPL": 2,
    "INL": 3,
    "OPL": 4,
    "ONL": 5,
    "EZ": 6,
    "RPE": 7,
}


def cleanup_label(
    label: np.ndarray,
    filter_size: int = 6,
) -> np.ndarray:
    """Fill small holes and remove small components within manual annotations.

    Args:
        label: Manual annotation.
        filter_size: Filter components with less or equal number of pixels.

    Returns:
        edited annotations
    """
    unique_ids = np.unique(label)[1:]
    new_label = np.zeros(label.shape)
    for num, i in enumerate(unique_ids):
        mask = label == i
        mask[mask > 0] = 1

        # Fill holes
        mask_closed = binary_closing(mask, structure=np.ones((2, 2)), border_value=1, iterations=2)

        # Remove small components
        seg = label_binary(mask_closed)
        unique_label, sizes = np.unique(seg, return_counts=True)
        for s, label_id in zip(sizes, unique_label):
            if s <= filter_size:
                seg[seg == label_id] = 0

        # Avoid overlapping segmentation instances
        seg[seg > 0] = num + 1
        new_label += seg
        new_label[new_label > num + 1] = num

    return new_label


def restrict_label_to_image(
    label: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    """Remove labels corresponding to image regions with zero intensity.
    The labels are removed for the largest connected component of zero intensity.

    Args:
        label: Manual annotations.
        image: Image data.

    Returns:
        edited annotations
    """
    zero_mask = image == 0
    zero_label = label_binary(zero_mask)
    unique_labels, sizes = np.unique(zero_label, return_counts=True)
    # remove component containing image intensity
    zero_id_mask = unique_labels != 0
    unique_labels = unique_labels[zero_id_mask]
    sizes = sizes[zero_id_mask]

    # get largest component of zeros
    if len(sizes) == 0:
        return label
    sizes, unique_label = zip(*sorted(zip(sizes, unique_labels), reverse=True))
    mask = zero_label == unique_label[0]

    label[mask == 1] = 0
    return label


def assign_layer_id(
    label: np.ndarray,
) -> np.ndarray:
    layer_order = find_layer_order(label)
    if layer_order is None:
        print("No layer order could be identified")
        return None
    else:
        number_layers = len(layer_order)
        if number_layers in LAYER_NUMBER_DICT.keys():
            layer_names = LAYER_NUMBER_DICT[number_layers]
            layer_indexes = [LAYER_LABEL_DICT[lay] for lay in layer_names]
        else:
            layer_indexes = [i + 1 for i in range(number_layers)]

        # re-label labels
        offset = max(layer_order)
        for i in layer_order:
            label[label == i] = i + offset

        for num, i in enumerate(layer_order):
            label[label == i + offset] = layer_indexes[num]

    return label
