import argparse
import os

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.measure import label as label_mask

from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from util import _derive_prompts_sam, _segment_from_prompts, _filter_prompts


def compute_iou_matrix(gt, pred):
    """
    gt, pred: labeled integer masks (0 = background)
    returns: IoU matrix (N_gt x N_pred)
    """
    gt_ids = np.unique(gt)[1:]
    pred_ids = np.unique(pred)[1:]

    iou = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)

    for i, g in enumerate(gt_ids):
        g_mask = gt == g
        g_area = g_mask.sum()

        for j, p in enumerate(pred_ids):
            p_mask = pred == p
            inter = np.logical_and(g_mask, p_mask).sum()
            union = g_area + p_mask.sum() - inter

            if union > 0:
                iou[i, j] = inter / union

    return iou


def match_instances_by_iou(iou, iou_threshold=0.5):
    """
    Hungarian matching with IoU threshold.
    """
    if iou.size == 0:
        return [], [], []

    cost = 1 - iou
    gt_idx, pred_idx = linear_sum_assignment(cost)

    matches = []
    for g, p in zip(gt_idx, pred_idx):
        if iou[g, p] >= iou_threshold:
            matches.append((g, p))

    matched_gt = set(g for g, _ in matches)
    matched_pred = set(p for _, p in matches)

    return matches, matched_gt, matched_pred


def compute_instance_metrics(gt_mask, pred_mask, iou_threshold=0.5):
    """
    Instance-level Precision, Recall, F1
    """

    gt_lab = label_mask(gt_mask)
    pred_lab = label_mask(pred_mask)

    iou = compute_iou_matrix(gt_lab, pred_lab)
    matches, matched_gt, matched_pred = match_instances_by_iou(
        iou, iou_threshold
    )

    n_gt = len(np.unique(gt_lab)) - 1
    n_pred = len(np.unique(pred_lab)) - 1

    TP = len(matches)
    FP = n_pred - TP
    FN = n_gt - TP

    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": TP,
        "fp": FP,
        "fn": FN,
    }


def eval_model_sam(input_dir, model_path):
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)

    h5_paths = [entry.path for entry in os.scandir(input_dir) if ".h5" in entry.name]
    h5_paths.sort()
    images = [np.array(h5py.File(p, 'r')["image"]) for p in h5_paths]
    labels = [np.array(h5py.File(p, 'r')["labels"]["original"]) for p in h5_paths]

    total_tp = total_fp = total_fn = 0
    for i, (image, label) in enumerate(zip(images, labels)):

        segmenter.initialize(image)
        foreground, boundary_distances = segmenter._foreground, segmenter._boundary_distances

        prompts = _derive_prompts_sam(foreground, boundary_distances)
        filtered_prompts = _filter_prompts(prompts)
        seg = _segment_from_prompts(predictor, image, filtered_prompts, min_size=150)

        # compare seg and labels
        metrics = compute_instance_metrics(label, seg)
        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]

        print(f"Image {i}: "
              f"P={metrics['precision']:.3f}, "
              f"R={metrics['recall']:.3f}, "
              f"F1={metrics['f1']:.3f}, "
              f"TP={metrics['tp']}, "
              f"FP={metrics['fp']}, "
              f"FN={metrics['fn']}")

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"Precision {precision}")
    print(f"Recall {recall}")
    print(f"F1-score {f1}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate micro-sam model on all images in a folder. Evaluates data in h5 format."
    )
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("--model", default="./oct-sam-v3.pt")
    args = parser.parse_args()

    eval_model_sam(args.input_dir, args.model,)


if __name__ == "__main__":
    main()
