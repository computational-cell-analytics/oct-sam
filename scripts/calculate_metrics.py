import argparse
from typing import List, Optional

import h5py
import imageio.v3 as imageio
import numpy as np

from oct_tools.metric_utils import get_etdrs_mask, run_measurement


def calculate_metrics(
    input_path: str,
    output_path: str,
    voxel_size: List[float],
    fovea_position: Optional[float] = None,
    reference_position: Optional[float] = None,
    etdrs_grid: Optional[str] = None,
):
    """Calculate metrics.

    Args:
        input_path: File path to 2D segmentation of OCT data.
        output_path: File path to save metrics as table in TSV format.
    """
    if ".h5" in input_path:
        h5_obj = h5py.File(input_path, "r")["labels"]
        if "edit_v3" in h5_obj.keys():
            print("Loading refined annotation edit_v3.")
            seg = np.array(h5_obj["edit_v3"])
        else:
            print("Loading original annotation.")
            seg = np.array(h5_obj["original"])
    else:
        seg = imageio.imread(input_path)
    if len(voxel_size) == 1:
        voxel_size = voxel_size * 2
    voxel_size = np.array(voxel_size)[::-1]

    fovea_point = None
    if fovea_position is not None:
        fovea_point = [0, fovea_position]

    reference_point = None
    if reference_position is not None:
        reference_point = [0, reference_position]

    tab = run_measurement(
        seg, spacing=voxel_size, extra_information=True,
        reference_point=reference_point,
        fovea_point=fovea_point,
    )
    if etdrs_grid is not None:
        if fovea_point is None:
            raise ValueError("You have to provide a fovea point to export an ETDRS grid.")
        etdrs_mask, notification_str = get_etdrs_mask(seg, tab, fovea_point=fovea_point)
        print(notification_str)
        imageio.imwrite(etdrs_grid, etdrs_mask)

    if ".tsv" in output_path:
        tab.to_csv(output_path, sep="\t", index=False)
    elif ".xlsx" in output_path:
        tab.to_excel(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate OCT-metrics for 2D segmentation. "
        "An ETDRS grid can optionally be created for a given fovea position."
    )
    parser.add_argument("-i", "--input", required=True, help="Input segmentation.")
    parser.add_argument("-o", "--output", required=True,
                        help="Output path. Supports 'tsv' and 'xlsx' as file extensions.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+",
                        default=[3.87166976, 5.8814],
                        help="Voxel size of input in micrometer.")
    parser.add_argument("--ref_position", type=float, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument("--fovea", type=float, default=None,
                        help="Position of fovea point on vertical axis for calculating area of ETDRS grid.")
    parser.add_argument("--etdrs_grid", type=str, default=None,
                        help="Export ETDRS grid.")

    args = parser.parse_args()

    calculate_metrics(
        args.input, args.output, args.voxel_size,
        fovea_position=args.fovea,
        reference_position=args.ref_position,
        etdrs_grid=args.etdrs_grid,
    )


if __name__ == "__main__":
    main()
