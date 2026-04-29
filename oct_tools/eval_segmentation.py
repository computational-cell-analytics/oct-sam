import json
import os
from typing import Optional

import h5py
import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from elf.evaluation import matching
from elf.evaluation.dice import symmetric_best_dice_score


def eval_segmentation_2d(
    data_dir: str,
    seg_dir: str,
    output_extension: str = "tif",
    check_nnunet: bool = False,
    label_key: str = "original",
    json_file: Optional[str] = None,
):
    """Evaluate predictions by comparing them to labels.
    The segmentation directory is searched for all matching names contained in the data directory.
    All predictions in the segmentation directory are compared to the labels of the data directory.

    Args:
        data_dir: Directory containing images and labels in H5 format.
        seg_dir: Directory containing segmentation in TIF, NIfTI, or another format.
        output_extension: Set output extension for segmentation data.
        check_nnunet: Check segmentation directory for nnU-Net output format.
        label_key: Key for H5 format for reference label data.
        json_file: File path to JSON dictionary storing the evaluated performance.
    """
    data_paths = [entry.path for entry in os.scandir(data_dir) if ".h5" in entry.name]
    data_paths.sort()
    labels = [np.array(h5py.File(p, "r")["labels"][label_key]) for p in data_paths]
    data_names = [os.path.splitext(os.path.basename(data_path))[0] for data_path in data_paths]

    segmentations = []
    # identify matching predictions
    if check_nnunet:
        output_extension = "nii.gz"
        for basename in data_names:
            # reference nnU-Net output format
            seg_path = os.path.join(seg_dir, f"oct_{basename}.{output_extension}")
            nib_data = nib.load(seg_path)
            seg = nib_data.get_fdata()
            seg = np.array(seg).astype("int64")
            segmentations.append(seg)

    else:
        for basename in data_names:
            seg_path = os.path.join(seg_dir, f"{basename}.{output_extension}")
            seg = imageio.imread(seg_path)
            segmentations.append(seg)

    precisions = []
    recalls = []
    f1s = []
    symm_dice_scores = []
    dict_list = []

    for (label, seg, basename) in zip(labels, segmentations, data_names):
        single_dict = {}
        single_dict["name"] = basename
        metrics = matching(seg, label)
        symm_dice = symmetric_best_dice_score(seg, label)

        msg = f"{basename}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        msg += f", DICE={symm_dice:.3f}"

        single_dict["precision"] = metrics['precision']
        single_dict["recall"] = metrics['recall']
        single_dict["f1"] = metrics['f1']

        print(msg)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        symm_dice_scores.append(symm_dice)
        dict_list.append(single_dict)

    precision = np.round(np.mean(precisions), 3)
    recall = np.round(np.mean(recalls), 3)
    f1 = np.round(np.mean(f1s), 3)
    symm_dice_score = np.round(np.mean(symm_dice_scores), 3)

    print("Overall precision:", precision)
    print("Overall recall:", recall)
    print("Overall f1-score:", f1)
    print("Overall symm-dice:", symm_dice_score)

    if json_file is not None:
        with open(json_file, "w") as f:
            json.dump(dict_list, f, indent='\t', separators=(',', ': '))
