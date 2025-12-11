import numpy as np
from skimage.measure import label
from skimage.measure import regionprops
from skimage.segmentation import watershed


def get_instance_stats(mask):
    """
    Returns a dict:
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


def find_anchor_segments(stats, width, tolerance=0.05):
    anchors = []
    full_ymax = width - 1
    margin = tolerance * width
    for sid, s in stats.items():
        if s['ymin'] <= margin and s['ymax'] >= full_ymax - margin:
            anchors.append(sid)
    # sort anchors from top to bottom
    anchors_xc = [stats[sid]["xc"] for sid in anchors]
    anchors_xc, anchors = zip(*sorted(zip(anchors_xc, anchors)))
    return anchors


def order_ids(seg, anchors):
    height, width = seg.shape
    id_relations = {}
    for a in anchors:
        id_relations[f"below_{a}"] = []
        id_relations[f"above_{a}"] = []

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
                        if col_ids.index(idx) > col_ids.index(a):
                            id_relations[f"below_{a}"].append(idx)
                        else:
                            id_relations[f"above_{a}"].append(idx)

    for a in anchors:
        id_relations[f"below_{a}"] = list(dict.fromkeys(id_relations[f"below_{a}"]))
        id_relations[f"above_{a}"] = list(dict.fromkeys(id_relations[f"above_{a}"]))

    # order relations into segments
    id_segments = {}
    for i in range(2 * len(anchors) + 1):
        id_segments[i] = []
        # anchor
        if i % 2 == 1:
            id_segments[i] = [anchors[i // 2]]

        # before first anchor
        elif i == 0:
            id_segments[0] = id_relations[f"above_{anchors[0]}"]

        # after last anchor
        elif i == 2 * len(anchors):
            id_segments[i] = id_relations[f"below_{anchors[-1]}"]

        else:
            anchor_above = anchors[(i - 1) // 2]
            anchor_below = anchors[(i + 1) // 2]
            id_segments[i] = list(set(id_relations[f"above_{anchor_below}"]).intersection(set(id_relations[f"below_{anchor_above}"])))

    filtered_segments = {}
    count = 0
    for stage, ids in id_segments.items():
        if len(ids) != 0:
            filtered_segments[count] = ids
            count += 1

    return filtered_segments


def next_horizontal(check_id, ids_sorted, stats, margin=25):
    if len(ids_sorted) == 1:
        return None
    for sid in ids_sorted:
        if check_id == sid:
            continue
        if (stats[check_id]["ymax"] + margin > stats[sid]["ymin"] and
            stats[check_id]["ymax"] < margin + stats[sid]["ymin"]):
            return sid
    return None


def find_matches(segment_dic, stats, horizontal_margin=25):
    group_ids = []
    for key, seg_ids in segment_dic.items():
        if len(seg_ids) == 1:
            group_ids.append(seg_ids)
        elif len(seg_ids) > 1:
            yc = [stats[sid]["yc"] for sid in seg_ids]
            yc, ids_sorted = zip(*sorted(zip(yc, seg_ids)))
            ids_sorted = list(ids_sorted)
            print(ids_sorted)
            ids_unmatched = ids_sorted.copy()
            for idx in ids_sorted:
                chain = []
                start_id = idx
                while start_id is not None:
                    chain.append(start_id)
                    next_id = next_horizontal(start_id, ids_unmatched, stats, margin=horizontal_margin)
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


def combine_matched_ids(seg, group_ids, stats):
    for num, gr in enumerate(group_ids):
        if len(gr) > 1:
            areas = [stats[sid]["area"] for sid in gr]
            new_id = gr[areas.index(max(areas))]
            for idx in gr:
                seg[seg == idx] = new_id
            group_ids[num] = [new_id]
    return seg, group_ids


def combine_horizontal(seg, tolerance=0.05):
    h, w = seg.shape

    stats = get_instance_stats(seg)

    ids = np.unique(seg)
    ids = ids[ids != 0]  # remove background
    anchors = []
    anchors = find_anchor_segments(stats, w, tolerance=tolerance)
    dic = order_ids(seg, anchors)
    group_ids = find_matches(dic, stats)
    seg, group_ids = combine_matched_ids(seg, group_ids, stats)
    return seg


def postprocess_segmentation(seg, img, min_thickness=5):
    seg = filter_min_thickness(seg, min_thickness=min_thickness)
    # seg = combine_horizontal(seg)
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
