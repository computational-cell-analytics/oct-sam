import numpy as np
from skimage.measure import label
from skimage.segmentation import watershed


def postprocess_segmentation(seg, img, min_thickness=5):
    seg = filter_min_thickness(seg, min_thickness=min_thickness)
    seg = fill_gaps_watershed(seg, img)
    return seg


def filter_min_thickness(
    seg: np.ndarray,
    min_thickness: int = 5,
) -> np.ndarray:
    """Filter out segmentation instances which are thinner than a given value.

    Args:
        seg: Integer labeled segmentation mask.
        min_thickness: Minimal thickness in px.

    Returns:
        Filtered segmentation.
    """
    ids = np.unique(seg)
    ids = ids[ids != 0]  # remove background
    height, width = seg.shape
    max_thickness = {}
    for idx in ids:
        max_thickness[idx] = 0
        for x in range(width):
            col = seg[:, x]
            seg_mask = (col == idx)
            max_thickness[idx] = max([max_thickness[idx], sum(seg_mask)])
    for (idx, thickness) in max_thickness.items():
        if thickness < min_thickness:
            print(f"Removed label {idx} because it is too thin with a thickness of {thickness}.")
            seg[seg == idx] = 0
    return seg


def fill_gaps_watershed(
    seg: np.ndarray,
    img: np.ndarray,
    max_label_size: int = 10000,
) -> np.ndarray:
    """Use morphological watershed to fill gaps between segmentation instances.

    Args:
        seg: Integer labeled segmentation mask.
        img: Grayscale image used as topography.
        max_label_size: Maximal label size. Instances with more occurences are filtered out.

    Returns:
        Filtered segmentation.
    """
    # Mask of unlabeled pixels
    gaps = seg == 0
    if not gaps.any():
        return seg

    # remove upper and lower gaps between segmentation and image borders
    gaps_label = label(gaps)
    seg_ids, sizes = np.unique(gaps_label, return_counts=True)
    sizes, seg_ids = zip(*sorted(zip(sizes, seg_ids), reverse=True))
    border_ids = [s for num, s in enumerate(seg_ids) if sizes[num] > max_label_size and s != 0]
    if len(border_ids) != 2:
        print(f"Length of border IDs: {len(border_ids)}.")
    for border_id in border_ids:
        gaps[gaps_label == border_id] = 0

    # Use segmentation labels as markers
    markers = seg.copy()

    # Watershed requires positive topography; invert if needed
    # lower intensity = deeper basin
    topography = img.max() - img

    filled = watershed(
        image=topography,
        markers=markers,
        mask=(seg > 0) | gaps
    )

    return filled
