import argparse
import os
from glob import glob

import h5py
import napari

from tqdm import tqdm


def _check_data(folder, label_key="labels/original"):
    files = glob(os.path.join(folder, "*.h5"))
    for ff in tqdm(files, desc=f"Checking {folder}"):
        with h5py.File(ff, "r") as f:
            image = f["image"][:]
            labels = f[label_key][:]
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        v.title = os.path.basename(ff)
        napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Check image and label data of H5 files in input folder with napari."
    )
    parser.add_argument("-i", "--input", required=True, help="Input folder.")

    args = parser.parse_args()
    _check_data(args.input)


if __name__ == "__main__":
    main()
