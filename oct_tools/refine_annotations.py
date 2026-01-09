import numpy as np
from scipy.ndimage import binary_closing
from skimage.measure import label as label_binary


def cleanup_label(
    label: np.ndarry,
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
    sizes, unique_label = zip(*sorted(zip(sizes, unique_labels), reverse=True))
    mask = zero_label == unique_label[0]

    label[mask == 1] = 0
    return label
