from typing import List, Optional, Tuple

import h5py
import imageio.v3 as imageio
import networkx as nx
import numpy as np
import pandas as pd

from skimage.measure import regionprops
from skimage.morphology import skeletonize

# This is the information about the voxel / pixel size extracted from one
# tomogram in eyepy. The unit is micrometer. This has to be double checked!
VOXEL_SIZE = (121.41720950603485, 3.87166976, 5.8814)


def _compute_thickness_per_column(
    mask: np.ndarray,
    spacing: Tuple[float],
) -> List[float]:
    """Compute thickness values per column for binary mask.

    Args:
        mask: Binary segmentation mask.
        spacing: Pixel spacing.

    Returns:
        List of thicknesses per column. Only shows values if segmentation mask is present in a column.
    """
    height, width = mask.shape
    thicknesses = []
    for idx in [1]:
        for x in range(width):
            col = mask[:, x]
            seg_mask = (col == idx)
            thicknesses.append(sum(seg_mask) * spacing[0])

    return thicknesses


def _thickness_at_reference(
    mask: np.ndarray,
    reference_position: float,
    spacing: Tuple[float],
) -> float:
    """Calculate thickness of binary mask at reference point.

    Args:
        mask: Binary segmentation mask.
        reference_point: Reference position of horizontal axis.
        spacing: Spacing between pixels.

    Returns:
        Thickness in physical values [µm].
    """
    col = mask[:, round(reference_position)]
    seg_mask = (col == 1)
    return sum(seg_mask) * spacing[0]


def _get_etdrs_grid_single(
    mask: np.ndarray,
    reference_position: float,
    spacing: Tuple[float],
    min_dist: int = 0,
    max_dist: int = 1,
) -> float:
    """Calculate area of the circular mapping of a 2D ETDRS grid for a binary mask.
    The position of the fovea is given as the reference poisition on the horizontal axis.
    The sections of the ETDRS grid are calculated as the segments between
    the minimal and maximal distances in relation to the reference point.

    Args:
        mask: Binary mask.
        reference_position: Position of the fovea on the horizontal axis.
        spacing: Pixel spacing of the 2D B-scan.
        min_dist: Distance to fovea position of inner limit of ETDRS area.
        max_dist: Distance to fovea position of outer limit of ETDRS area.

    Returns:
        Area of section in µm².
    """
    central_y = round(reference_position)
    lower_limit = 0
    upper_limit = mask.shape[1] - 1

    start_o = max([central_y - round(max_dist / spacing[1]), lower_limit])
    end_o = min([central_y + round(max_dist / spacing[1]), upper_limit])

    if min_dist != 0:
        start_i = max([central_y - round(min_dist / spacing[1]), lower_limit])
        end_i = min([central_y + round(min_dist / spacing[1]), upper_limit])
        col = [o for o in range(start_o, end_o + 1) if o not in [i for i in range(start_i, end_i + 1)]]

    else:
        col = [o for o in range(start_o, end_o + 1)]

    num_cols = mask.shape[1]
    col_mask = np.zeros(num_cols, dtype=bool)

    # Set the specified row indices to True, if they are within bounds
    for idx in col:
        if 0 <= idx < num_cols:
            col_mask[idx] = True

    # Broadcast the 1D boolean mask to 2D
    bool_mask = col_mask[np.newaxis, :] * np.ones_like(mask, dtype=bool)

    # Apply the boolean mask to the original binary mask
    filtered_mask = mask * bool_mask
    return np.count_nonzero(filtered_mask) * spacing[0] * spacing[1], filtered_mask


def get_columns(
    mask: np.ndarray,
    reference_position: float,
    spacing: Tuple[float],
    min_dist: int = 0,
    max_dist: int = 1,
):
    """Get list of columns based on distance to reference point.
    """
    central_y = round(reference_position)
    lower_limit = 0
    upper_limit = mask.shape[1] - 1

    start_o = max([central_y - round(max_dist / spacing[1]), lower_limit])
    end_o = min([central_y + round(max_dist / spacing[1]), upper_limit])

    if min_dist != 0:
        start_i = max([central_y - round(min_dist / spacing[1]), lower_limit])
        end_i = min([central_y + round(min_dist / spacing[1]), upper_limit])
        col = [o for o in range(start_o, end_o + 1) if o not in [i for i in range(start_i, end_i + 1)]]

    else:
        col = [o for o in range(start_o, end_o + 1)]

    return col


def _get_etdrs_grid_all(mask, reference_point, spacing):
    """Get areas of central etdrs grid for binary mask.
    """
    # central: 0-0.5 mm
    area_c, mask_c = _get_etdrs_grid_single(mask, reference_point[1], spacing, 0, 500)

    # inner ring: 0.5 - 1.5 mm
    area_i, mask_i = _get_etdrs_grid_single(mask, reference_point[1], spacing, 500, 1500)

    # outer ring: 1.5 - 3 mm
    area_o, mask_o = _get_etdrs_grid_single(mask, reference_point[1], spacing, 1500, 3000)
    return area_c, area_i, area_o, mask_c, mask_i, mask_o


def _skel_coords(skel: np.ndarray) -> np.ndarray:
    """Return (row, col) coordinates of foreground skeleton pixels."""
    return np.column_stack(np.nonzero(skel))


def _build_skeleton_graph(coords: np.ndarray, pixel_spacing):
    """
    Build an 8-neighborhood graph on skeleton pixels with physical edge weights.
    """
    sy, sx = pixel_spacing
    index_of = {tuple(p): i for i, p in enumerate(map(tuple, coords))}
    G = nx.Graph()

    for i, (r, c) in enumerate(coords):
        G.add_node(i, pos=(int(r), int(c)))
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nbr = (r + dr, c + dc)
                j = index_of.get(nbr)
                if j is None:
                    continue
                # Undirected; let NetworkX handle duplicate edge attempts
                w = float(np.hypot(dr * sy, dc * sx))
                G.add_edge(i, j, weight=w)
    return G


def _longest_geodesic_path(G: nx.Graph):
    """
    Among all endpoint pairs (degree == 1), return the weighted-longest
    shortest path. If no endpoints exist (e.g., small loop), use a double sweep.
    """
    if len(G) == 0:
        return [], 0.0

    degrees = dict(G.degree)
    endpoints = [n for n, d in degrees.items() if d == 1]

    def _path_len(path_nodes):
        return sum(G[a][b]["weight"] for a, b in zip(path_nodes[:-1], path_nodes[1:]))

    best_nodes, best_len = [], 0.0

    if len(endpoints) >= 2:
        # Try all endpoint pairs; for long thin objects this is typically tiny.
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                s, t = endpoints[i], endpoints[j]
                try:
                    nodes = nx.shortest_path(G, s, t, weight="weight")
                except nx.NetworkXNoPath:
                    continue
                L = _path_len(nodes)
                if L > best_len:
                    best_nodes, best_len = nodes, L
    else:
        # Fallback: double sweep on the largest connected component
        comp = max(nx.connected_components(G), key=len)
        sub = G.subgraph(comp).copy()
        n0 = next(iter(sub.nodes))
        far1 = max(
            nx.single_source_dijkstra_path_length(sub, n0, weight="weight").items(),
            key=lambda x: x[1],
        )[0]
        far2 = max(
            nx.single_source_dijkstra_path_length(sub, far1, weight="weight").items(),
            key=lambda x: x[1],
        )[0]
        best_nodes = nx.shortest_path(sub, far1, far2, weight="weight")
        best_len = _path_len(best_nodes)

    return best_nodes, best_len


def _compute_length(mask, pixel_spacing=(1.0, 1.0)):
    skel = skeletonize(mask.astype(bool))
    coords = _skel_coords(skel)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int), 0.0

    G = _build_skeleton_graph(coords, pixel_spacing)
    nodes, length = _longest_geodesic_path(G)
    if not nodes:
        length = 0
    # polyline = np.array([G.nodes[n]["pos"] for n in nodes], dtype=int)
    return float(length)


def get_etdrs_mask(
    segmentation: np.ndarray,
    measurement: Optional[dict] = None,
    spacing: Optional[Tuple[float]] = None,
    fovea_point: Optional[Tuple[float]] = None,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Obtain a mask which visualizes the sections of the central ETDRS
    (Early Treatment Diabetic Retinopathy Study) grid of the OCT diagram.

    Args:
        segmentation: Segmentation mask.
        measurement: Measurement dictionary obtained from run_measurement.
        spacing: Voxel size.
        fovea_point: Foveal reference point for the calculation of the ETDRS areas.

    Returns:
        Mask showing the different sections of the ETDRS grid.
        Notification showing the central fovea thickness.
    """
    if fovea_point is None:
        return None, None

    if spacing is None:
        spacing = VOXEL_SIZE[1:]  # Get the pixel spacing in micrometer.

    if measurement is None:
        measurement = run_measurement(segmentation, spacing=spacing, fovea_point=fovea_point)

    unit = "µm"
    # calculate overlay for ETDRS area
    mask = (segmentation != 0)
    area_c, area_i, area_o, mask_c, mask_i, mask_o = _get_etdrs_grid_all(mask, fovea_point, spacing)
    etdrs_mask = mask_c + 2 * mask_i + 3 * mask_o

    central_foveal_thickness = sum(measurement[f"CFT@{fovea_point[1]}px[{unit}]"])
    notification_str = f"The central foveal thickness is {round(central_foveal_thickness, 2)} {unit}."
    return etdrs_mask, notification_str


def run_measurement(
    segmentation: np.ndarray,
    spacing: Optional[Tuple[float]] = None,
    extra_information: bool = False,
    extra_columns: Optional[pd.DataFrame] = None,
    fovea_point: Optional[Tuple[float]] = None,
    reference_point: Optional[Tuple[float]] = None,
) -> pd.DataFrame:
    """Calculate measurements for OCT tailored metrics.

    Args:
        segmentation: 2D OCT segmentation.
        spacing: Voxel size.
        extra_information: Add additional information about layer length, max, min, and mean thickness.
        extra_columns: Additional columns to store with the dataset.
        fovea_point: Foveal reference point for the calculation of the ETDRS areas.
        reference_point: Reference point for calculating the thickness of every slice.

    Returns:
        Measurement values as dataframe.
    """
    if spacing is None:
        spacing = VOXEL_SIZE[1:]  # Get the pixel spacing in micrometer.

    unit = "µm"
    unit_area = "mm"
    factor_area = 1 / 1000000
    props = regionprops(segmentation, spacing=spacing)
    measurement = {
        "label_id": [],
        f"area[{unit_area}²]": [],
    }

    if extra_information:
        measurement[f"length[{unit}]"] = []
        measurement[f"max_thickness[{unit}]"] = []
        measurement[f"min_thickness[{unit}]"] = []
        measurement[f"mean_thickness[{unit}]"] = []
        measurement[f"stdev_thickness[{unit}]"] = []

    if reference_point is not None:
        print(f"ref point {reference_point}")
        if reference_point[0] > segmentation.shape[0] or reference_point[1] > segmentation.shape[1]:
            raise ValueError(f"Reference point {reference_point} does not lie within segmentation boundary.")
        measurement[f"thickness@{reference_point[1]}px[{unit}]"] = []

    if fovea_point is not None:
        measurement[f"CFT@{fovea_point[1]}px[{unit}]"] = []
        measurement[f"central_area[{unit_area}²]"] = []
        measurement[f"inner_ring[{unit_area}²]"] = []
        measurement[f"outer_ring[{unit_area}²]"] = []

    for prop in props:
        measurement["label_id"].append(prop.label)
        measurement[f"area[{unit_area}²]"].append(prop.area * factor_area)
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:2], prop.bbox[2:]))
        mask = (segmentation[bb] == prop.label)
        mask_all = (segmentation == prop.label)

        if extra_information:
            # Compute the centerline to measure the length.
            length = _compute_length(mask, spacing)
            measurement[f"length[{unit}]"].append(length)

            # Compute the layer thickness by distance from upper to lower boundary.
            # This computes the thickness across each point for both the upper and lower boundary.
            thickness = _compute_thickness_per_column(mask_all, spacing)
            measurement[f"max_thickness[{unit}]"].append(max(thickness))
            measurement[f"min_thickness[{unit}]"].append(min(thickness))
            measurement[f"mean_thickness[{unit}]"].append(np.mean(thickness))
            measurement[f"stdev_thickness[{unit}]"].append(np.std(thickness))

        if fovea_point is not None:
            central_thickness = _thickness_at_reference(mask_all, fovea_point[1], spacing)
            measurement[f"CFT@{fovea_point[1]}px[{unit}]"].append(central_thickness)

            area_c, area_i, area_o, _, _, _ = _get_etdrs_grid_all(mask_all, fovea_point, spacing)
            measurement[f"central_area[{unit_area}²]"].append(area_c * factor_area)
            measurement[f"inner_ring[{unit_area}²]"].append(area_i * factor_area)
            measurement[f"outer_ring[{unit_area}²]"].append(area_o * factor_area)

        if reference_point is not None:
            thickness_at_ref = _thickness_at_reference(mask_all, reference_point[1], spacing)
            measurement[f"thickness@{reference_point[1]}px[{unit}]"].append(thickness_at_ref)

    measurement = pd.DataFrame(measurement)
    if extra_columns is not None:
        measurement = pd.merge(measurement, extra_columns, on="label_id", how="outer")
    return measurement


def calculate_metrics(
    input_path: str,
    output_path: Optional[str],
    voxel_size: List[float],
    fovea_position: Optional[float] = None,
    reference_position: Optional[float] = None,
    etdrs_grid: Optional[str] = None,
):
    """Calculate metrics for semgentation/label data.
    The input is either label data in H5 format.

    Args:
        input_path: File path to 2D segmentation of OCT data.
        output_path: File path to save metrics as table in TSV format.
    """
    if ".h5" in input_path:
        h5_obj = h5py.File(input_path, "r")["labels"]
        # look for refined annotations
        if "edit_v3" in h5_obj.keys():
            print("Loading refined annotation edit_v3.")
            seg = np.array(h5_obj["edit_v3"])
        # use original annotations
        else:
            print("Loading original annotation.")
            seg = np.array(h5_obj["original"])
    else:
        seg = imageio.imread(input_path)

    if len(voxel_size) == 1:
        voxel_size = voxel_size * 2
    voxel_size = np.array(voxel_size)[::-1]

    fovea_point = None
    if fovea_position is not None:
        fovea_point = [0, fovea_position]

    reference_point = None
    if reference_position is not None:
        reference_point = [0, reference_position]

    tab = run_measurement(
        seg, spacing=voxel_size, extra_information=True,
        reference_point=reference_point,
        fovea_point=fovea_point,
    )
    if etdrs_grid is not None:
        if fovea_point is None:
            raise ValueError("You have to provide the foveal point to export an ETDRS grid.")
        etdrs_mask, notification_str = get_etdrs_mask(seg, tab, fovea_point=fovea_point)
        print(notification_str)
        imageio.imwrite(etdrs_grid, etdrs_mask)

    if output_path is None:
        print(tab)
    else:
        if ".tsv" in output_path:
            tab.to_csv(output_path, sep="\t", index=False)
        elif ".xlsx" in output_path:
            tab.to_excel(output_path, index=False)
