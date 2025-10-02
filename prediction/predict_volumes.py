import os
from glob import glob

import imageio.v3 as imageio
import numpy as np

ROOT = "/mnt/lustre-grete/usr/u12086/data/oct/data_20250625"


def predict_volume(model, volume):
    import torch_em
    from torch_em.util.prediction import predict_with_padding

    foreground, boundaries = [], []
    for section in volume:
        input_ = torch_em.transform.raw.standardize(section)
        pred = predict_with_padding(model, input_, min_divisible=(16, 16))
        foreground.append(pred[0:1, 0])
        boundaries.append(pred[0:1, 1])
    foreground = np.concatenate(foreground, axis=0)
    boundaries = np.concatenate(boundaries, axis=0)
    return foreground, boundaries


def predict_volumes():
    import torch_em
    from tqdm import tqdm

    files = glob(os.path.join(ROOT, "converted", "*.tif"))

    fg_folder = os.path.join(ROOT, "foreground-predictions")
    os.makedirs(fg_folder, exist_ok=True)
    bd_folder = os.path.join(ROOT, "boundary-predictions")
    os.makedirs(bd_folder, exist_ok=True)

    model = torch_em.util.load_model("../training/checkpoints/oct-boundary-foreground")

    for ff in tqdm(files):
        volume = imageio.imread(ff)
        foreground, boundaries = predict_volume(model, volume)
        assert volume.shape == foreground.shape
        assert volume.shape == boundaries.shape
        imageio.imwrite(os.path.join(fg_folder, os.path.basename(ff)), foreground)
        imageio.imwrite(os.path.join(bd_folder, os.path.basename(ff)), boundaries)


def check_predictions():
    import napari

    files = glob(os.path.join(ROOT, "converted", "*.tif"))
    for ff in files:

        volume = imageio.imread(ff)

        fname = os.path.basename(ff)
        foreground = imageio.imread(os.path.join(ROOT, "foreground-predictions", fname))
        boundaries = imageio.imread(os.path.join(ROOT, "boundary-predictions", fname))

        v = napari.Viewer()
        v.add_image(volume)
        v.add_image(foreground)
        v.add_image(boundaries)
        napari.run()


def main():
    # predict_volumes()
    check_predictions()


if __name__ == "__main__":
    main()
