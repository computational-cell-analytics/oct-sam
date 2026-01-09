from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import label, regionprops
from skimage.segmentation import watershed


def get_instance_stats(
    mask: np.ndarray,
) -> dict:
    """ Creates a dictionary with measurements for a segmentation mask.

    Args:
        mask: Segmentation labels.

    Returns:
        Dictionary for every index:
            id -> {
                'xmin', 'xmax', 'ymin', 'ymax', 'xc', 'yc',
                'height', 'width', 'coords', 'area'
            }
    """
    props = regionprops(mask)
    stats = {}

    for p in props:
        stats[p.label] = {
            'xmin': p.bbox[0],
            'ymin': p.bbox[1],
            'xmax': p.bbox[2] - 1,
            'ymax': p.bbox[3] - 1,
            'xc': (p.bbox[0] + p.bbox[2] - 1) / 2,
            'yc': (p.bbox[1] + p.bbox[3] - 1) / 2,
            'coords': p.coords,
            'area': p.area,
        }

    return stats


def find_anchor_segments(
    stats: dict,
    width: int,
    tolerance: float = 0.05,
) -> List[int]:
    """Find segmentation instances, which span the whole y-dimension.
    These are referred to as "anchors" and act as references to other segmentation IDs
    which may be broken and could be merged.

    Args:
        stats: Measurements for each segmentation id.
        width: Width of segmentation.
        tolerance: Margin of error for segmentations which do not span the whole length.

    Returns:
        List of indexes for segmentation instances which span the whole y-dimension.
            They can act as a reference for the order of layers in x-dimension.
    """
    anchors = []
    full_ymax = width - 1
    margin = tolerance * width
    for sid, s in stats.items():
        if s['ymin'] <= margin and s['ymax'] >= full_ymax - margin:
            anchors.append(sid)

    # sort anchors from top to bottom
    if len(anchors) != 0:
        anchors_xc = [stats[sid]["xc"] for sid in anchors]
        anchors_xc, anchors = zip(*sorted(zip(anchors_xc, anchors)))

    return anchors


def most_frequent_index(lst: List[int]) -> int:
    """Return the most frequent index in a list.

    Args:
        lst: List containing integers.

    Returns:
        Most frequent element of list.
    """
    counts = {}
    for num in lst:
        counts[num] = counts.get(num, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def order_ids_for_offset(
    ids: List[int],
    offset_dic: dict,
) -> dict:
    """Order indexes of (potentially) broken segmentation instances according to their offset.
    The offset dictionary is given for a specific anchor.

    Args:
        ids: List of indexes for a group of segmentation instances.
    """
    order_dic = {}
    for idx in ids:
        offset = offset_dic[idx]
        if offset in list(order_dic.keys()):
            order_dic[offset].append(idx)
        else:
            order_dic[offset] = [idx]
    return order_dic


def group_ids_into_layers(
    seg: np.ndarray,
    anchors: List[int],
) -> dict:
    """Group segmentation IDs into different layers.
    The division into groups is based on the position in x-dimension
    relative to the anchor layers.
    1) Create dictionary assigning offset of other IDs in relation to anchor layers.
    2) Take most frequent offset value.
    3) Order IDs from top to bottom into layers.
    4) Re-assign keys and remove empty layers.
    5) Group IDs with same offset into nested lists.

    Args:
        seg: Segmentation.
        anchors: List of IDs of "anchor" segmentation instances which cover the whole y-dimension.

    Returns:
        Dictionary of filtered segments.
    """
    height, width = seg.shape
    id_relations = {}
    for a in anchors:
        id_relations[a] = {}

    # 1) Assign offset to other indexes based on relation to anchor layers.
    for x in range(width):
        col = seg[:, x]
        col = list(dict.fromkeys(list(col)))
        col_ids = [c for c in col if c != 0]
        for idx in col_ids:
            if idx in anchors:
                pass
            else:
                for a in anchors:
                    if a in col_ids:
                        offset = col_ids.index(idx) - col_ids.index(a)
                        if idx in list(id_relations[a].keys()):
                            id_relations[a][idx].append(offset)
                        else:
                            id_relations[a][idx] = [offset]

    # 2) Set most frequent offset as value.
    for a in anchors:
        for key in id_relations[a].keys():
            id_relations[a][key] = most_frequent_index(id_relations[a][key])

    # 3) Order IDs into layers.
    id_segments = {}
    for i in range(2 * len(anchors) + 1):
        id_segments[i] = []
        # anchor layer
        if i % 2 == 1:
            id_segments[i] = [anchors[i // 2]]

        # before first anchor layer
        elif i == 0:
            id_segments[0] = [k for k in id_relations[anchors[0]].keys() if id_relations[anchors[0]][k] < 0]

        # after last anchor layer
        elif i == 2 * len(anchors):
            id_segments[i] = [k for k in id_relations[anchors[-1]].keys() if id_relations[anchors[-1]][k] > 0]

        # between two anchor layers
        else:
            anchor_above = anchors[(i - 1) // 2]
            anchor_below = anchors[(i + 1) // 2]
            ids_below_upper = [k for k in id_relations[anchor_above].keys() if id_relations[anchor_above][k] > 0]
            ids_above_lower = [k for k in id_relations[anchor_below].keys() if id_relations[anchor_below][k] < 0]
            id_segments[i] = list(set(ids_above_lower).intersection(set(ids_below_upper)))

    # 4) Re-assign keys and remove empty layers.
    filtered_layers = {}
    count = 0
    for stage, ids in id_segments.items():
        if len(ids) != 0:
            filtered_layers[count] = ids
            count += 1

    # 5) Group IDs with same offset into nested lists.
    for (key, ids) in filtered_layers.items():
        if len(ids) > 1:
            order_dic = order_ids_for_offset(ids, id_relations[anchors[0]])
            ordered_list = [order_dic[k] for k in order_dic.keys()]
            filtered_layers[key] = ordered_list

    return filtered_layers


def flatten_layer_dic(layer_dic: dict) -> List[List[int]]:
    """Flatten dictionary containing layers with IDs.
    Nested lists will be individual groups.

    Args:
        layer_dic: Dictionary containing IDs for different layers.

    Returns:

    :param segment_dic: Description
    :type segment_dic: dict
    :return: Description
    :rtype: List[List[int]]
    """
    group_ids = []
    for key, seg_group in layer_dic.items():
        if len(seg_group) == 1:
            if isinstance(seg_group[0], list):
                group_ids.extend(seg_group)
            else:
                group_ids.append(seg_group)
        else:
            for group in seg_group:
                group_ids.append(group)
    return group_ids


def next_horizontal(
    check_id: int,
    ids_sorted: List[int],
    stats: dict,
    margin: int = 10,
) -> Optional[int]:
    """Identify next horizontal ID by checking position in y-dimension.

    Args:
        check_id: Segmentation ID to be matched.
        ids_sorted: List of IDs sorted by central y-position.
        stats: Measurement dictionary for segmentation IDs.
        margin: Margin in pixels for overlap.

    Returns:
        ID of matched segmentation.
    """
    if len(ids_sorted) == 1:
        return None
    for sid in ids_sorted:
        if check_id == sid:
            continue
        if (stats[check_id]["ymax"] + margin > stats[sid]["ymin"] and
                stats[check_id]["ymax"] < margin + stats[sid]["ymin"]):
            return sid
    return None


def match_layers_horizontally(
    layer_dic: dict,
    stats: dict,
) -> List[List[int]]:
    """Match segmentation IDs within a layer based on horizontal position.

    Args:
        layer_dic: Dictionary containing segmentation IDs. Keys represent the order from top to bottom.
        stats: Measurements for segmentation IDs.

    Returns:
        Nested list for grouped IDs. All segmentation instances for IDs within a nested list will be combined.
    """
    group_ids = []
    for key, seg_group in layer_dic.items():
        if len(seg_group) == 1 and not isinstance(seg_group[0], list):
            group_ids.append(seg_group)
        elif len(seg_group) != 0:
            seg_ids = [x for xs in seg_group for x in xs]
            yc = [stats[sid]["yc"] for sid in seg_ids]
            yc, ids_sorted = zip(*sorted(zip(yc, seg_ids)))
            ids_sorted = list(ids_sorted)
            ids_unmatched = ids_sorted.copy()
            for idx in ids_sorted:
                chain = []
                start_id = idx
                while start_id is not None:
                    chain.append(start_id)
                    next_id = next_horizontal(start_id, ids_unmatched, stats)
                    if next_id is not None:
                        ids_unmatched.remove(start_id)
                    elif len(chain) > 1:
                        print("connected", chain)
                        group_ids.append(chain)
                        ids_unmatched.remove(start_id)
                    start_id = next_id

            for idx in ids_sorted:
                if not any(idx in gr for gr in group_ids):
                    group_ids.append([idx])

    return group_ids


def combine_grouped_ids(
    seg: np.ndarray,
    group_ids: List[List[int]],
    stats: dict,
) -> Tuple[np.ndarray, List[List[int]]]:
    """Combine grouped IDs to have the same segmentation ID.
    The segmentation ID of the largest component is kept.

    Args:
        seg: Segmentation.
        group_ids: List of segmentation IDs.
        stats: Measurements for segmentation IDs.

    Returns:
        Segmentation with combined instances.
        Grouped IDs with updated IDs after merging.
    """
    for num, gr in enumerate(group_ids):
        if len(gr) > 1:
            areas = [stats[sid]["area"] for sid in gr]
            new_id = gr[areas.index(max(areas))]
            for idx in gr:
                seg[seg == idx] = new_id
            group_ids[num] = [new_id]
    return seg, group_ids


def merge_segmentation_horizontally(
    seg: np.ndarray,
    anchor_tolerance: float = 0.05,
    matching_method: str = "offset",
) -> np.ndarray:
    """Check segmentation IDs for disconnected instances.
    Combine multiple instances in the same horizontal layer.

    Args:
        seg: Input segmentation.
        anchor_tolerance: Tolerance for the length of anchor layers, which span the entire y-dimension.
        matching_method: Method for matching disconnected segmentation IDs. Either "offset" or "y_position".

    Returns:
        Segmentation with combined IDs.
    """
    h, w = seg.shape

    stats = get_instance_stats(seg)
    ids = np.unique(seg)
    ids = ids[ids != 0]  # remove background
    anchors = []
    anchors = find_anchor_segments(stats, w, tolerance=anchor_tolerance)
    if len(anchors) == 0:
        print("Could not find any anchor layer. Skipping horizontal merging.")
        return seg

    layer_dic = group_ids_into_layers(seg, anchors)

    if matching_method == "y_position":
        group_ids = match_layers_horizontally(layer_dic, stats)
    elif matching_method == "offset":
        group_ids = flatten_layer_dic(layer_dic)
    else:
        raise ValueError("Choose either 'offset' or 'y_position' as matching_method.")

    seg, group_ids = combine_grouped_ids(seg, group_ids, stats)
    return seg


def postprocess_segmentation(
    seg: np.ndarray,
    img: np.ndarray,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
    min_thickness: int = 5,
    matching_method: str = "offset",
    verbose: bool = True,
) -> np.ndarray:
    """Post-process segmentation iteratively with multiple functions.
    The order and selection of the functions are determined by the parameter "postprocess_functions".
    "merge_horizontal": Merge disconnected segmentation instances along horizontal layers.
    "filter_thin": Filter segmentation instances which are thinner than a given minimal pixel value.
    "fill_gaps": Fill gaps within holes not connected to the upper or lower background via watershed.

    Args:
        seg: Segmentation.
        img: Image.
        postprocessing_functions: List of functions. Post-processing will be performed in the given order.
        min_thickness: Minimal thickness of layers for "filter_thin"-method.
        matching_method: Method for matching disconnected segmentation IDs for "merge_horizontal"-method.
            Either "offset" or "y_position".
        verbose: Whether to print post-processing info.

    Returns:
        Post-processed segmentation.
    """
    # Define the mapping of postprocessing methods to their corresponding functions and parameters
    method_map = {
        "merge_horizontal": {
            "func": merge_segmentation_horizontally,
            "requires_img": False,
            "params": {"matching_method": matching_method}
        },
        "filter_thin": {
            "func": filter_min_thickness,
            "requires_img": False,
            "params": {"min_thickness": min_thickness}
        },
        "fill_gaps": {
            "func": fill_gaps_watershed,
            "requires_img": True,
            "params": {}
        }
        # Add new methods here as needed
    }

    if any(element in ["no", "No", "none", "None"] for element in postprocess_functions):
        print("No post-processing.")
        return seg

    for method in postprocess_functions:
        if method in method_map:
            method_config = method_map[method]
            func = method_config["func"]
            requires_img = method_config["requires_img"]
            params = method_config["params"]

            # Print a message for logging
            if verbose:
                print(f"Applying post-processing method: {method}")

            # Call the function with the appropriate arguments
            if requires_img:
                seg = func(seg, img, **params)
            else:
                seg = func(seg, **params)
        else:
            print(f"Warning: Unknown post-processing method '{method}'.")

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
    for border_id in border_ids:
        gaps[gaps_label == border_id] = 0

    # Use segmentation labels as markers
    markers = seg.copy()
    markers = markers.astype(int)

    # Watershed requires positive topography; invert if needed
    # lower intensity = deeper basin
    topography = img.max() - img

    filled = watershed(
        image=topography,
        markers=markers,
        mask=(seg > 0) | gaps
    )

    return filled
