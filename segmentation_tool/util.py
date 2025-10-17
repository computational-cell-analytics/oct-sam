import os
from collections import deque
from typing import Tuple

import imageio.v3 as imageio
import networkx as nx
import numpy as np
import pandas as pd
import torch
import vigra

from tqdm import tqdm
from unet import UNet2d
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from skimage.segmentation import find_boundaries

try:
    import eyepy as ep
except ImportError:
    ep = None

# This is the informaiton about the voxel / pixel size extracted from one
# tomogram in eyepy. The unit is millimeteter. This has to be double checked!
VOXEL_SIZE = (0.12141720950603485, 0.0038716697599738836, 0.0056914291344583035)


def standardize(raw: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    raw = raw.astype("float32")
    mean = raw.mean(keepdims=True)
    raw -= mean
    std = raw.std(keepdims=True)
    raw /= (std + eps)
    return raw


def predict_with_padding(
    model: torch.nn.Module,
    input_: np.ndarray,
    min_divisible: Tuple[int, ...],
    with_channels: bool = False,
) -> np.ndarray:
    if with_channels:
        assert len(min_divisible) + 1 == input_.ndim, f"{min_divisible}, {input_.ndim}"
        min_divisible_ = (1,) + min_divisible
    else:
        assert len(min_divisible) == input_.ndim
        min_divisible_ = min_divisible

    if any(sh % md != 0 for sh, md in zip(input_.shape, min_divisible_)):
        pad_width = tuple(
            (0, 0 if sh % md == 0 else md - sh % md)
            for sh, md in zip(input_.shape, min_divisible_)
        )
        crop_padding = tuple(slice(0, sh) for sh in input_.shape)
        input_ = np.pad(input_, pad_width, mode="reflect")
    else:
        crop_padding = None

    ndim = input_.ndim
    ndim_model = 1 + ndim if with_channels else 2 + ndim

    device = next(model.parameters()).device

    expand_dim = (None,) * (ndim_model - ndim)
    with torch.no_grad():
        model_input = torch.from_numpy(input_[expand_dim]).to(device)
        output = model(model_input)
        output = output.cpu().numpy()

    if crop_padding is not None:
        crop_padding = (slice(None),) * (output.ndim - len(crop_padding)) + crop_padding
        output = output[crop_padding]

    return output


def run_prediction(volume, model_path, out_channels=2):
    # Load the model.
    model_state = torch.load(model_path, weights_only=True, map_location="cpu")
    model = UNet2d(in_channels=1, out_channels=out_channels, initial_features=32, final_activation="Sigmoid")
    model.load_state_dict(model_state)

    prediction = []
    for section in tqdm(volume, desc="Run prediction"):
        input_ = standardize(section)
        pred = predict_with_padding(model, input_, min_divisible=(16, 16))
        pred = pred.transpose((1, 0, 2, 3))
        prediction.append(pred)

    prediction = np.concatenate(prediction, axis=1)
    return prediction


def load_volume(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in (".tif", ".tiff"):
        vol = imageio.imread(path)
    elif ext == ".vol":
        vol = ep.import_heyex_vol(path).data
    else:
        raise ValueError(f"Can't load filetype with ending: {ext}")
    return vol


def _sliding_max_1d(a: np.ndarray, window: int) -> np.ndarray:
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    r = window // 2
    b = np.pad(a, (r, r), mode="edge")
    q = deque()
    out = np.empty_like(a)
    for i, val in enumerate(b):
        while q and q[-1][0] <= val:
            q.pop()
        q.append((val, i))
        if i >= window - 1:
            start = i - (window - 1)
            while q[0][1] < start:
                q.popleft()
            out[start] = q[0][0]
    return out


def normalize_rows_sliding_max(img: np.ndarray, window: int, eps: float = 1e-9) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    row_max = img.max(axis=1)
    denom = _sliding_max_1d(row_max, window)
    denom = np.maximum(denom, eps)
    return img / denom[:, None]


def normalize_sliding_max_2d(img: np.ndarray, window_y: int, window_x: int, eps: float = 1e-12) -> np.ndarray:
    if img.ndim != 2:
        raise ValueError("img must be a 2D array")
    if window_y < 1 or window_y % 2 == 0 or window_x < 1 or window_x % 2 == 0:
        raise ValueError("window_y and window_x must be positive odd integers")
    H, W = img.shape
    tmp = np.empty_like(img)
    for y in range(H):
        tmp[y, :] = _sliding_max_1d(img[y, :], window_x)
    maxf = np.empty_like(img)
    for x in range(W):
        maxf[:, x] = _sliding_max_1d(tmp[:, x], window_y)
    denom = np.maximum(maxf.astype(np.float64), eps)
    return img.astype(np.float64) / denom


def get_edge_ratio(rag, labels):
    import nifty.graph.rag as nrag

    edge_builder = nrag.ragCoordinates(rag)
    edge_values = np.arange(1, rag.numberOfEdges + 1)
    edge_image = edge_builder.edgesToVolume(edge_values, edgeDirection=0).astype("uint32")

    uv_ids = rag.uvIds()
    edge_props = regionprops(edge_image)

    edge_ratios = np.zeros(rag.numberOfEdges, dtype="float32")
    edge_areas = np.zeros(rag.numberOfEdges, dtype="float32")
    for prop in edge_props:
        edge_id = prop.label - 1
        len_x = float(prop.bbox[3] - prop.bbox[1])
        len_y = float(prop.bbox[2] - prop.bbox[0])
        if 0 in uv_ids[edge_id] or len_y < 5:  # don't merge to background or too tiny edges
            ratio = 1
        else:
            ratio = min(len_x / len_y, 1)
        edge_ratios[edge_id] = ratio
        edge_areas[edge_id] = len_y

    # import napari
    # import nifty.graph.rag as nrag
    # edge_builder = nrag.ragCoordinates(rag)
    # edge_image = edge_builder.edgesToVolume(edge_ratios, edgeDirection=0)
    # v = napari.Viewer()
    # v.add_image(edge_image)
    # v.add_labels(labels)
    # napari.run()

    return edge_ratios


def get_edge_ydist(rag, labels):
    uv_ids = rag.uvIds()
    props = regionprops(labels)

    centroids_y = np.zeros(rag.numberOfNodes, dtype="float32")
    for prop in props:
        node_id = prop.label
        centroids_y[node_id] = prop.centroid[1]

    edge_features = np.zeros(rag.numberOfEdges, dtype="float32")
    for edge_id in range(rag.numberOfEdges):
        u, v = uv_ids[edge_id]
        if u == 0 or v == 0:
            continue
        ydist = np.abs(centroids_y[u] - centroids_y[v])
        edge_features[edge_id] = ydist

    # edge_features /= edge_features.max()
    edge_features[(uv_ids == 0).any(axis=1)] = edge_features.max()

    import napari
    import nifty.graph.rag as nrag
    edge_builder = nrag.ragCoordinates(rag)
    # edge_values = np.arange(1, rag.numberOfEdges + 1)
    edge_image = edge_builder.edgesToVolume(edge_features, edgeDirection=1)
    v = napari.Viewer()
    v.add_image(edge_image)
    v.add_labels(labels)
    napari.run()

    return edge_features


def merge_overseg(labels, directed_dist, beta=0.5):
    # from elf.segmentation.features import compute_rag, project_node_labels_to_pixels, compute_boundary_features
    from elf.segmentation.features import compute_rag, project_node_labels_to_pixels
    from elf.segmentation.multicut import compute_edge_costs, multicut_kernighan_lin

    rag = compute_rag(labels)
    edge_features = get_edge_ratio(rag, labels)
    # edge_features = get_edge_ydist(rag, labels)
    # edge_features = compute_boundary_features(rag, (1. - directed_dist).astype("float32"))
    # edge_features = edge_features[:, 2]

    costs = compute_edge_costs(edge_features, beta=beta)
    node_labels = multicut_kernighan_lin(rag, costs)
    seg = project_node_labels_to_pixels(rag, node_labels)
    return seg


def _compute_thickness(mask, spacing):
    boundaries = find_boundaries(mask, mode="outer")
    boundary_components = label(boundaries)

    # Determine the largest two boundary components and assign them
    # to the upper / lower boundary.
    props = regionprops(boundary_components)
    ids = np.array([prop.label for prop in props])
    sizes = np.array([prop.area for prop in props])
    heights = np.array([prop.centroid[0] for prop in props])

    two_largest = ids[np.argsort(sizes)[::-1][:2]]
    h1, h2 = heights[two_largest - 1]
    upper_id, lower_id = two_largest if h1 < h2 else two_largest[::-1]
    upper_mask, lower_mask = boundary_components == upper_id, boundary_components == lower_id

    # TODO spacig in the distance trafos
    distance_to_upper = vigra.filters.vectorDistanceTransform(upper_mask.astype("float32"))
    distance_to_upper = np.abs(distance_to_upper[..., 0])
    lower_thickness = distance_to_upper[lower_mask]

    distance_to_lower = vigra.filters.vectorDistanceTransform(lower_mask.astype("float32"))
    distance_to_lower = np.abs(distance_to_lower[..., 0])
    upper_thickness = distance_to_lower[upper_mask]

    # import napari
    # v = napari.Viewer()
    # v.add_image(mask)
    # v.add_image(distance_to_upper)
    # v.add_labels(upper_mask)
    # v.add_labels(lower_mask)
    # napari.run()

    return upper_thickness, lower_thickness


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


def run_measurement(segmentation, spacing=None):
    if spacing is None:
        spacing = VOXEL_SIZE[1:]  # Get the pixel spacing in millimeter.

    props = regionprops(segmentation, spacing=spacing)
    measurement = {
        "label_id": [],
        "area": [],
        "length": [],
        "max_thickness": [],
        "min_thickness": [],
        "mean_thickness": [],
        "stdev_thickness": [],
    }
    for prop in props:
        measurement["label_id"].append(prop.label)
        measurement["area"].append(prop.area)
        bb = tuple(slice(start, stop) for start, stop in zip(prop.bbox[:2], prop.bbox[2:]))
        mask = (segmentation[bb] == prop.label)

        # Compute the centerline to measure the length.
        length = _compute_length(mask, spacing)
        measurement["length"].append(length)

        # Compute the layer thickness by distance from upper to lower boundary.
        # This computes the thickness across each point for both the upper and lower boundary.
        # We then compute statistics over this.
        # Later, we also want to estimate the thickness at certain radii.
        upper_thickness, lower_thickness = _compute_thickness(mask, spacing)
        thickness = np.concatenate([upper_thickness, lower_thickness], axis=0)
        max_thickness, min_thickness = thickness.max(), thickness.min()
        mean_thickness, stdev_thickness = thickness.mean(), thickness.std()
        measurement["max_thickness"].append(max_thickness)
        measurement["min_thickness"].append(min_thickness)
        measurement["mean_thickness"].append(mean_thickness)
        measurement["stdev_thickness"].append(stdev_thickness)

    measurement = pd.DataFrame(measurement)
    return measurement
