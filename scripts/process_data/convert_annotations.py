import argparse
import os
from glob import glob

import h5py

from oct_tools.data.data_reader import read_bscan


def main():
    parser = argparse.ArgumentParser(
        description="Convert the annotations of a training cycle into H5 files with the datasets 'image' and "
        "'labels/original'. Each label '<name>_committed_objects.tif' needs the image '<name>.tif'."
    )
    parser.add_argument("-i", "--input", required=True, help="Input folder with the images and the labels.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")

    args = parser.parse_args()

    label_paths = sorted(glob(os.path.join(args.input, "*_committed_objects.tif")))
    if not label_paths:
        raise ValueError(f"{args.input} has no files '*_committed_objects.tif'.")
    os.makedirs(args.output, exist_ok=True)

    for label_path in label_paths:
        image_path = label_path[:-len("_committed_objects.tif")] + ".tif"
        name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(args.output, f"{name}.h5")
        # Later steps add the edited labels to the same file, so the script does not replace it.
        if os.path.exists(output_path):
            print(f"Skipping {output_path}. The file exists.")
            continue
        with h5py.File(output_path, "w") as f:
            f.create_dataset("image", data=read_bscan(image_path), compression="gzip")
            f.create_dataset("labels/original", data=read_bscan(label_path), compression="gzip")


if __name__ == "__main__":
    main()
