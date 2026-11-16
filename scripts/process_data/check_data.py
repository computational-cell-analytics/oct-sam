import argparse
import os
from glob import glob

import napari
from tqdm import tqdm

from oct_tools.data.data_reader import read_bscan


def main():
    parser = argparse.ArgumentParser(
        description="Show the image and the label of each H5 file in a folder with napari."
    )
    parser.add_argument("-i", "--input", required=True, help="Input folder.")
    parser.add_argument("-k", "--label_key", type=str, default="labels/original",
                        help="Label dataset in the H5 files, e.g. 'labels/edit_v3', or 'masks' for the data "
                        "of prepare_data.py. Default: 'labels/original'.")

    args = parser.parse_args()

    for path in tqdm(sorted(glob(os.path.join(args.input, "*.h5"))), desc=f"Checking {args.input}"):
        v = napari.Viewer()
        v.add_image(read_bscan(path))
        v.add_labels(read_bscan(path, keys=(args.label_key,)))
        v.title = os.path.basename(path)
        napari.run()


if __name__ == "__main__":
    main()
