from collections import deque

import numpy as np
import torch
import vigra

import segment_anything.utils.amg as amg_utils
from micro_sam._vendored import batched_mask_to_box, mask_to_rle_pytorch
from micro_sam.inference import batched_inference
from skimage.measure import label, regionprops
from skimage.segmentation import relabel_sequential
from torchvision.ops.boxes import batched_nms
from torch_em.model.unet import UNet2d
from torch_em.transform.raw import standardize
from torch_em.util.prediction import predict_with_padding


#
# TODO: refactor to micro-sam.
# Functionality for non-maximum suppression.
# The NMS for masks is taken from CellSeg1:
# https://github.com/Nuisal/cellseg1/blob/1c027c2568b83494d2662d1fbecec9aafb478ee0/mask_nms.py
#


def rle_to_mask(rle):
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx:idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()


def overlap_matrix(boxes):
    x1 = torch.max(boxes[:, None, 0], boxes[:, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[:, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[:, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    return (w * h) > 0


def calculate_ious_between_pred_masks(masks, boxes, diagonal_value=1):
    masks = (
        masks.detach() if isinstance(masks, torch.Tensor) else torch.tensor(masks)
    )
    n_points = masks.shape[0]
    m = torch.zeros((n_points, n_points))

    overlap_m = overlap_matrix(boxes)

    for i in range(n_points):
        js = torch.where(overlap_m[i])[0]
        js_half = js[js > i]

        if len(js_half) > 0:
            intersection = torch.logical_and(masks[i], masks[js_half]).sum(dim=(1, 2))
            union = torch.logical_or(masks[i], masks[js_half]).sum(dim=(1, 2))
            iou = intersection / union
            m[i, js_half] = iou

    m = m + m.T
    m.fill_diagonal_(diagonal_value)
    return m


def calculate_scores(iou_preds, stability_score):
    return iou_preds * stability_score


def batched_mask_nms(rles, boxes, scores, nms_thresh):
    if len(rles) == 0:
        return torch.tensor([], dtype=torch.int64)

    masks = torch.stack([torch.tensor(rle_to_mask(rle)) for rle in rles])
    boxes = (
        boxes.detach()
        if isinstance(boxes, torch.Tensor)
        else torch.tensor(boxes)
    )
    scores = (
        scores.detach()
        if isinstance(scores, torch.Tensor)
        else torch.tensor(scores)
    )

    iou_matrix = calculate_ious_between_pred_masks(masks, boxes)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        if len(sorted_indices) == 1:
            break

        iou_values = iou_matrix[i, sorted_indices[1:]]
        mask = iou_values <= nms_thresh
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep)


def apply_nms(predictions, shape, min_size, perform_box_nms=False, nms_thresh=0.9):
    data = amg_utils.MaskData(
        masks=torch.cat([pred["segmentation"][None] for pred in predictions], dim=0),
        iou_preds=torch.tensor([pred["predicted_iou"] for pred in predictions]),
    )
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    data["boxes"] = batched_mask_to_box(data["masks"])
    data["area"] = [mask.sum() for mask in data["masks"]]

    if min_size > 0:
        keep_by_size = torch.tensor([i for i, area in enumerate(data["area"]) if area > min_size])
        data.filter(keep_by_size)

    if perform_box_nms:
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=nms_thresh,
        )
    else:
        keep_by_nms = batched_mask_nms(
            rles=data["rles"],
            boxes=data["boxes"].float(),
            scores=data["iou_preds"],
            nms_thresh=nms_thresh,
        )
    data.filter(keep_by_nms)

    mask_data = [
        {"segmentation": mask, "area": area} for mask, area in zip(data["masks"], data["area"])
    ]
    segmentation = mask_data_to_segmentation(mask_data, min_object_size=min_size)
    return segmentation


#
# Other utility functionality.
#


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


def mask_data_to_segmentation(masks, min_object_size):
    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    shape = next(iter(masks))["segmentation"].shape
    segmentation = np.zeros(shape, dtype="uint32")

    def require_numpy(mask):
        return mask.cpu().numpy() if torch.is_tensor(mask) else mask

    for seg_id, mask in enumerate(masks, 1):
        this_mask = require_numpy(mask["segmentation"])
        this_mask = np.logical_and(this_mask, segmentation == 0)
        segmentation[this_mask] = seg_id

    segmentation = label(segmentation)
    seg_ids, sizes = np.unique(segmentation, return_counts=True)
    filter_ids = seg_ids[sizes < min_object_size]

    segmentation[np.isin(segmentation, filter_ids)] = 0
    segmentation = relabel_sequential(segmentation)[0]
    return segmentation


def _derive_prompts(model, image, seed_threshold=0.6, return_pred=False):
    input_ = standardize(image)
    pred = predict_with_padding(model, input_, min_divisible=(16, 16))
    pred = pred.squeeze()[:2]

    foreground, boundaries = pred[0], pred[1]
    mask = foreground > 0.5
    bd_mask = boundaries > 0.5

    directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
    directed_dist[~mask] = 0
    directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
    directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

    seeds = label(directed_dist > seed_threshold)
    props = regionprops(seeds)
    prompts = np.array([prop.centroid for prop in props])
    if return_pred:
        return prompts, pred
    return prompts


def _derive_prompts_sam(foreground, boundary_distances, seed_threshold=0.6):
    # Find the largest foregorund piece, we only keep the prompts in there.
    mask = label(foreground > 0.5)
    ids, sizes = np.unique(mask, return_counts=True)
    ids, sizes = ids[1:], sizes[1:]
    mask = mask == ids[np.argmax(sizes)]

    # Get the boundary mask and compute the distances for seeding.
    bd_mask = boundary_distances > 0.5

    directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
    directed_dist[~mask] = 0
    directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
    directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

    seeds = label(directed_dist > seed_threshold)
    props = regionprops(seeds)
    prompts = np.array([prop.centroid for prop in props])
    return prompts


def _segment_from_prompts(predictor, image, prompts, min_size):
    points = prompts[:, None, ::-1]
    labels = np.ones((len(prompts), 1))
    predictions = batched_inference(
        predictor, image, batch_size=16, points=points, point_labels=labels, return_instance_segmentation=False
    )
    segmentation = apply_nms(predictions, image.shape, min_size=min_size, perform_box_nms=False)
    return segmentation


def _load_model(model_path):
    model_state = torch.load(model_path, weights_only=True, map_location="cpu")
    model = UNet2d(in_channels=1, out_channels=4, initial_features=32, final_activation="Sigmoid")
    model.load_state_dict(model_state)
    return model
