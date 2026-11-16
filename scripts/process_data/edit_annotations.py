import argparse
import json
import os
from typing import Optional

import h5py
import numpy as np

from oct_tools.data.data_reader import read_bscan
from oct_tools.postprocessing.postprocessing import fill_gaps_watershed
from oct_tools.data.refine_annotations import assign_layer_id, cleanup_label, restrict_label_to_image

# edit_v2 and edit_v3 have the same steps. edit_v3 is the label for training.
EDIT_STEPS = {
    "edit_v1": ("cleanup", "fill_gaps", "restrict"),
    "edit_v2": ("cleanup", "fill_gaps", "restrict", "assign_layer_id"),
    "edit_v3": ("cleanup", "fill_gaps", "restrict", "assign_layer_id"),
}


def edit_annotations(input_dir: str, output_dir: Optional[str] = None, label_version: str = "edit_v1"):
    """Edit the manual annotations in "labels/original" and write them to "labels/<label_version>".

    The steps fill holes, remove isolated fragments, fill the gaps between the layers and restrict the
    labels to the image area. edit_v2 and edit_v3 also assign the layer IDs 1-7. If no column of a label
    contains every layer, the layer IDs are unknown and the file gets no "labels/<label_version>".
    An existing "labels/<label_version>" is not replaced.
    The file `layer_overview.json` in the output folder lists the files for each number of layers in the
    original annotation.

    Args:
        input_dir: Folder with the H5 files.
        output_dir: Output folder. By default, the function edits the input files.
        label_version: "edit_v1", "edit_v2" or "edit_v3".
    """
    steps = EDIT_STEPS[label_version]
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    h5_names = sorted(entry.name for entry in os.scandir(input_dir) if entry.name.lower().endswith(".h5"))
    layer_dict = {}
    skipped = []

    for h5_name in h5_names:
        path = os.path.join(input_dir, h5_name)
        image = read_bscan(path)
        label = read_bscan(path, keys=("labels/original",))

        layer_dict.setdefault(len(np.unique(label)[1:]), []).append(h5_name)

        if "cleanup" in steps:
            label = cleanup_label(label)
        if "fill_gaps" in steps:
            label = fill_gaps_watershed(label, image)
        if "restrict" in steps:
            label = restrict_label_to_image(label, image)
        if "assign_layer_id" in steps:
            label = assign_layer_id(label)
            if label is None:
                print(f"Skipping {h5_name}. No column contains every layer.")
                skipped.append(h5_name)
                continue

        with h5py.File(os.path.join(output_dir, h5_name), "a") as f:
            if f"labels/{label_version}" not in f:
                f.create_dataset(f"labels/{label_version}", data=label, compression="gzip")

    layer_dict = {key: sorted(names) for key, names in sorted(layer_dict.items())}
    with open(os.path.join(output_dir, "layer_overview.json"), "w") as f:
        json.dump(layer_dict, f, indent='\t', separators=(',', ': '))
    if skipped:
        print(f"Warning: {len(skipped)} files have no labels/{label_version}: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Edit the manual annotations of H5 files: clean up the labels, fill the gaps between "
        "the layers, restrict them to the image area and, from edit_v2, assign the layer IDs 1-7."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Folder with H5 files that contain 'image' and 'labels/original'.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output folder. Default: Edit the files in place.")
    parser.add_argument("-v", "--version", type=str, default="edit_v1", choices=list(EDIT_STEPS),
                        help="Label version, written to 'labels/<version>'. Default: 'edit_v1'.")

    args = parser.parse_args()

    edit_annotations(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_version=args.version,
    )


if __name__ == "__main__":
    main()
