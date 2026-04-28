import argparse
import os
from typing import Optional

import imageio.v3 as imageio
import nibabel as nib
import numpy as np


def convert_nifti_to_tif(
    input_folder: str,
    output_folder: Optional[str],
    label_data: bool
):
    file_paths = [entry.path for entry in os.scandir(input_folder) if ".nii.gz" in entry.name]
    file_paths.sort()

    if output_folder is None:
        output_folder = input_folder

    for ff in file_paths:
        base_name = os.path.basename(ff)
        base_name = base_name.split(".nii.gz")[0]
        out_path = os.path.join(output_folder, f"{base_name}.tif")
        my_img = nib.load(ff)
        arr = my_img.get_fdata()
        if label_data:
            arr = np.array(arr).astype("uint32")
        imageio.imwrite(out_path, arr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI inference data to TIF format."
    )

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory containing NIfTI files.")
    parser.add_argument("-o", "--output", type=str,
                        help="Output directory for TIF files. Default: Same as input directory.")
    parser.add_argument("--label", action="store_true",
                        help="Create label data.")

    args = parser.parse_args()

    convert_nifti_to_tif(
        input_folder=args.input,
        output_folder=args.output,
        label_data=args.label,
    )


if __name__ == "__main__":
    main()
