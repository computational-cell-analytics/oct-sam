import argparse
import os
from typing import Optional

import h5py
import numpy as np

from oct_tools.refine_annotations import cleanup_label, restrict_label_to_image
from oct_tools.postprocessing import fill_gaps_watershed


def edit_annotations(
    input_dir: str,
    output_dir: Optional[str] = None,
    label_version: str = "edit_v1",
):
    """Edit manual annotations by filling holes, removing isolated fragments,
    filling gaps between horizontal layers, and restricting the labels to the image area.

    Args:
        input_dir: Input directory containing image and label data in h5 format.
        output_dir: Output directory for edited annotations.
        label_version: String for labeling edited annotations in h5 output.
    """
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    h5_paths = [entry.path for entry in os.scandir(input_dir) if ".h5" in entry.name]
    h5_paths.sort()
    h5_names = [entry.name for entry in os.scandir(input_dir) if ".h5" in entry.name]
    h5_names.sort()
    images = [np.array(h5py.File(p, "r")["image"]) for p in h5_paths]
    labels = [np.array(h5py.File(p, "r")["labels"]["original"]) for p in h5_paths]

    for h5_name, image, label in zip(h5_names, images, labels):
        label = cleanup_label(label)
        label = fill_gaps_watershed(label, image)
        label = restrict_label_to_image(label, image)

        output_path = os.path.join(output_dir, h5_name)
        with h5py.File(output_path, "a") as f:
            f.create_dataset(f"labels/{label_version}", data=label, compression="gzip")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation performance on all images in a folder. "
        "Evaluates image and label data in h5 format and segmentation data in TIF format."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory containing images and labels in h5 format.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Directory containing edited annotations. Default: Edit files in place.")
    parser.add_argument("-v", "--version", type=str, default="edit_v1",
                        help="Description for data path of output in h5 file. Default: 'edit_v1'.")

    args = parser.parse_args()

    edit_annotations(
        args.input_dir, args.output_dir,
    )


if __name__ == "__main__":
    main()
