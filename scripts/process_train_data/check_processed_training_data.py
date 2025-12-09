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


def check_initial_training_data():
    _check_data("../data/data_20250619_resaved/")


def check_more_initial_training_data():
    _check_data("../data/data_20250717_resaved")


def check_cycle1():
    _check_data("../data/training_data_cycle_1")


def main():
    # check_initial_training_data()
    check_more_initial_training_data()
    # check_cycle1()


if __name__ == "__main__":
    main()
