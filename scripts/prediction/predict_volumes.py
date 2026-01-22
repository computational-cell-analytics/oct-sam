import argparse
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np


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


def predict_volumes(model_path, input_dir, output_dir):
    import torch_em
    from tqdm import tqdm

    files = glob(os.path.join(input_dir, "*.tif"))

    fg_folder = os.path.join(output_dir, "foreground-predictions")
    os.makedirs(fg_folder, exist_ok=True)
    bd_folder = os.path.join(output_dir, "boundary-predictions")
    os.makedirs(bd_folder, exist_ok=True)

    model = torch_em.util.load_model(model_path)

    for ff in tqdm(files):
        volume = imageio.imread(ff)
        print(volume.shape)
        foreground, boundaries = predict_volume(model, volume)
        assert volume.shape == foreground.shape
        assert volume.shape == boundaries.shape
        imageio.imwrite(os.path.join(fg_folder, os.path.basename(ff)), foreground)
        imageio.imwrite(os.path.join(bd_folder, os.path.basename(ff)), boundaries)


def check_predictions(image_dir, output_dir):
    import napari

    files = glob(os.path.join(image_dir, "*.tif"))
    for ff in files:

        volume = imageio.imread(ff)

        fname = os.path.basename(ff)
        foreground = imageio.imread(os.path.join(output_dir, "foreground-predictions", fname))
        boundaries = imageio.imread(os.path.join(output_dir, "boundary-predictions", fname))

        v = napari.Viewer()
        v.add_image(volume)
        v.add_image(foreground)
        v.add_image(boundaries)
        napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Apply SAM model on a single or multiple slices of input data."
    )
    parser.add_argument("-i", "--input", required=True, help="Input directory with image data in tif format.")
    parser.add_argument("-o", "--output", required=True, help="Output data for predictions.")
    parser.add_argument("-m", "--model", required=True, help="Input model.")
    parser.add_argument("-c", "--check", action="store_true", help="Check prediction with napari.")

    args = parser.parse_args()
    predict_volumes(args.model, args.input, args.output)
    if args.check:
        check_predictions(args.input, args.output)


if __name__ == "__main__":
    main()
