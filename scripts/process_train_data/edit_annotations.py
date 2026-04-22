import argparse
import json
import os
from typing import Optional

import h5py
import numpy as np

from oct_tools.refine_annotations import cleanup_label, restrict_label_to_image, assign_layer_id
from oct_tools.postprocessing import fill_gaps_watershed

# edit_v1: Label clean up, Fill gaps, Restrict to image
# edit_v2: Label clean up, Fill gaps, Restrict to image, common class IDs


def edit_annotations(
    input_dir: str,
    output_dir: Optional[str] = None,
    label_version: str = "edit_v1",
    cleanup_label_flag: bool = False,
    fill_gaps_watershed_flag: bool = False,
    restrict_label_to_image_flag: bool = False,
    assign_class_id_flag: bool = False,
    layer_overview: str = "layer_overview.json",
):
    """Edit manual annotations by filling holes, removing isolated fragments,
    filling gaps between horizontal layers, and restricting the labels to the image area.

    Args:
        input_dir: Input directory containing image and label data in H5 format.
        output_dir: Output directory for edited annotations.
        label_version: String for labeling edited annotations in H5 output.
        assign_class_id: Flag for assigning common class id.
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

    if label_version == "edit_v1":
        print(f"Using pre-defined editing for label version {label_version}.")
        cleanup_label_flag = True
        fill_gaps_watershed_flag = True
        restrict_label_to_image_flag = True
        assign_class_id_flag = False

    if label_version in ["edit_v2", "edit_v3"]:
        print(f"Using pre-defined editing for label version {label_version}.")
        cleanup_label_flag = True
        fill_gaps_watershed_flag = True
        restrict_label_to_image_flag = True
        assign_class_id_flag = True

    layer_dict = {}

    for h5_name, image, label in zip(h5_names, images, labels):
        number_layers = len(np.unique(label)[1:])

        if number_layers not in list(layer_dict.keys()):
            layer_dict[number_layers] = []
        layer_dict[number_layers].append(h5_name)

        if cleanup_label_flag:
            label = cleanup_label(label)

        if fill_gaps_watershed_flag:
            label = fill_gaps_watershed(label, image)

        if restrict_label_to_image_flag:
            label = restrict_label_to_image(label, image)

        if assign_class_id_flag:
            label = assign_layer_id(label)

        output_path = os.path.join(output_dir, h5_name)
        with h5py.File(output_path, "a") as f:
            if f"labels/{label_version}" not in f:
                f.create_dataset(f"labels/{label_version}", data=label, compression="gzip")

    for key in layer_dict.keys():
        layer_dict[key].sort()
    layer_dict = dict(sorted(layer_dict.items()))

    with open(layer_overview, "w") as f:
        json.dump(layer_dict, f, indent='\t', separators=(',', ': '))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation performance on all images in a folder. "
        "Evaluates image and label data in H5 format and segmentation data in TIF format."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Directory containing images and labels in H5 format.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Directory containing edited annotations. Default: Edit files in place.")
    parser.add_argument("-v", "--version", type=str, default="edit_v1",
                        help="Description for data path of output in H5 file. Default: 'edit_v1'.")

    args = parser.parse_args()

    edit_annotations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_version=args.version,
    )


if __name__ == "__main__":
    main()
