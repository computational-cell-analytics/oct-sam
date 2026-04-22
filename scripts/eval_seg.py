import argparse
import os

import h5py
import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from elf.evaluation import matching
from elf.evaluation.dice import symmetric_best_dice_score


def eval_segmentation(
    img_dir: str,
    seg_dir: str,
    output_extension: str = "tif",
    check_nnunet: bool = False,
    label_key: str = "original",
):
    h5_paths = [entry.path for entry in os.scandir(img_dir) if ".h5" in entry.name]
    h5_paths.sort()
    images = [np.array(h5py.File(p, "r")["image"]) for p in h5_paths]
    labels = [np.array(h5py.File(p, "r")["labels"][label_key]) for p in h5_paths]

    segmentations = []

    if check_nnunet:
        output_extension = "nii.gz"
        for h5_path in h5_paths:
            basename = "".join(os.path.basename(h5_path).split(".")[:-1])
            seg_path = os.path.join(seg_dir, f"oct_{basename}.{output_extension}")
            nib_data = nib.load(seg_path)
            seg = nib_data.get_fdata()
            seg = np.array(seg).astype("int64")
            segmentations.append(seg)

    else:
        for h5_path in h5_paths:
            basename = "".join(os.path.basename(h5_path).split(".")[:-1])
            seg_path = os.path.join(seg_dir, f"{basename}.{output_extension}")
            seg = imageio.imread(seg_path)
            segmentations.append(seg)

    precisions, recalls, f1s = [], [], []
    symm_dice_scores = []
    for i, (image, label, seg) in enumerate(zip(images, labels, segmentations)):
        h5_path = h5_paths[i]
        fname = os.path.basename(h5_path)
        metrics = matching(seg, label)
        symm_dice = symmetric_best_dice_score(seg, label)

        msg = f"Image {fname}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        msg += f", DICE={symm_dice:.3f}"
        print(msg)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        symm_dice_scores.append(symm_dice)

    precision = np.round(np.mean(precisions), 3)
    recall = np.round(np.mean(recalls), 3)
    f1 = np.round(np.mean(f1s), 3)
    symm_dice_score = np.round(np.mean(symm_dice_scores), 3)

    print("Overall precision:", precision)
    print("Overall recall:", recall)
    print("Overall f1-score:", f1)
    print("Overall symm-dice:", symm_dice_score)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation performance on all images in a folder. "
        "Evaluates image and label data in H5 format and segmentation data in TIF format."
    )
    parser.add_argument("-i", "--img_dir", type=str, required=True,
                        help="Directory containing images and labels in H5 format.")
    parser.add_argument("-s", "--seg_dir", type=str, required=True,
                        help="Directory containing segmentation in TIF format.")
    parser.add_argument("--nnunet", action="store_true",
                        help="Check for nnU-Net inference format.")
    parser.add_argument("--label_key", type=str, default="original",
                        help="Key for labels stored in H5 format.")

    args = parser.parse_args()

    eval_segmentation(
        args.img_dir,
        args.seg_dir,
        check_nnunet=args.nnunet,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()
