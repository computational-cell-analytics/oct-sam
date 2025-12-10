import argparse
from typing import List

import h5py
import imageio.v3 as imageio
import numpy as np

from oct_tools.segmentation_utils import run_measurement


def calculate_metrics(
    input_path: str,
    output_path: str,
    voxel_size: List[float],
):
    """Calculate metrics.

    Args:
        input_path: File path to 2D segmentation of OCT data.
        output_path: File path to save metrics as table in TSV format.
    """
    if ".h5" in input_path:
        seg = np.array(h5py.File(input_path, "r")["labels"]["original"])
    else:
        seg = imageio.imread(input_path)
    if len(voxel_size) == 1:
        voxel_size = voxel_size * 2
    voxel_size = np.array(voxel_size)[::-1]
    tab = run_measurement(seg, spacing=voxel_size)
    if ".tsv" in output_path:
        tab.to_csv(output_path, sep="\t", index=False)
    elif ".xlsx" in output_path:
        tab.to_excel(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate oct-metrics for 2D segmentation."
    )
    parser.add_argument("-i", "--input", required=True, help="Input segmentation.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path. Supports 'tsv' and 'xlsx' as file extensions.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+",
                        default=[0.0038716697599738836, 0.0056914291344583035],
                        help="Voxel size of input in millimeter.")

    args = parser.parse_args()

    calculate_metrics(
        args.input, args.output, args.voxel_size
    )


if __name__ == "__main__":
    main()
