import argparse
import os

import imageio.v3 as imageio
import numpy as np
import vigra

from skimage.measure import label
from skimage.segmentation import watershed
from oct_tools.segmentation_utils import normalize_sliding_max_2d, merge_overseg


def _run_segmentation(foreground, boundaries, seed_threshold: float = 0.6, merge: bool = False):
    mask = foreground > 0.5
    bd_mask = boundaries > 0.5

    directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
    directed_dist[~mask] = 0
    directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
    directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

    seeds = label(directed_dist > seed_threshold)
    over_seg = watershed(1. - directed_dist, markers=seeds, mask=mask)

    if merge:
        seg = merge_overseg(over_seg, directed_dist, beta=0.5)
    else:
        seg = over_seg
    return seg


def segmentation_unet(input_dir, segmentation_dir=None):
    if segmentation_dir is None:
        segmentation_dir = os.path.join(input_dir, "segmentation")
    boundary_dir = os.path.join(input_dir, "boundary-predictions")
    foreground_dir = os.path.join(input_dir, "foreground-predictions")

    boundary_files = [entry.path for entry in os.scandir(boundary_dir) if ".tif" in entry.name]
    foreground_files = [entry.path for entry in os.scandir(foreground_dir) if ".tif" in entry.name]
    boundary_files.sort()
    foreground_files.sort()

    assert len(boundary_files) == len(foreground_files)

    for boundary_file, foreground_file in zip(boundary_files, foreground_files):
        basename = ".".join(os.path.basename(boundary_file).split(".")[:-1])
        print(basename)
        boundary = imageio.imread(boundary_file)
        boundary = boundary[0, :]
        foreground = imageio.imread(foreground_file)
        foreground = foreground[0, :]
        seg = _run_segmentation(foreground, boundary)
        output_seg = os.path.join(segmentation_dir, f"{basename}.tif")
        imageio.imwrite(output_seg, seg)


def main():
    parser = argparse.ArgumentParser(description="First version of the OCT segmentation tool.")
    parser.add_argument(
        "-i", "--input_dir", required=True, help="The path to the oct file. Can either be a .tiff or .vol file."
    )
    parser.add_argument(
        "-p", "--prediction", default=None,
        help="Optional path to a precomputed network prediction. Must be a .tiff file."
    )
    args = parser.parse_args()

    segmentation_unet(args.input_dir, args.prediction)


if __name__ == "__main__":
    main()
