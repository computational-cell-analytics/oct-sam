import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari

from tqdm import tqdm


def check_training_data(root):

    images = sorted(glob(os.path.join(root, "**/*_cropped*.tif"), recursive=True))
    masks = sorted(glob(os.path.join(root, "**/*_masked*.tif"), recursive=True))
    assert len(images) == len(masks)

    print("Checking", len(images), "images")
    for x, y in zip(images, masks):
        x, y = imageio.imread(x), imageio.imread(y)
        v = napari.Viewer()
        v.add_image(x)
        v.add_labels(y)
        napari.run()


# TODO also map the labels of the annotations to the layer.
# For this we need to make use of the respective layer indications in the table.
def resave_training_data(root, output):
    images = sorted(glob(os.path.join(root, "**/*_cropped*.tif"), recursive=True))
    masks = sorted(glob(os.path.join(root, "**/*_masked*.tif"), recursive=True))
    assert len(images) == len(masks)

    os.makedirs(output, exist_ok=True)

    for x, y in tqdm(zip(images, masks), total=len(images)):
        fname = os.path.basename(x)
        output_name = "_".join(fname.split("_")[:4]) + ".h5"

        x, y = imageio.imread(x), imageio.imread(y)
        # Crop away the scale bar.
        x, y = x[:-100, :], y[:-100, :]
        assert x.shape == y.shape

        output_path = os.path.join(output, output_name)
        with h5py.File(output_path, "a") as f:
            f.create_dataset("image", data=x, compression="gzip")
            f.create_dataset("labels/original", data=y, compression="gzip")


def main():
    # check_training_data("../data/data_20250619")
    resave_training_data("../data/data_20250619", "../data/data_20250619_resaved")


if __name__ == "__main__":
    main()
