import argparse
import os
from glob import glob
from typing import Tuple

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

FILTER_DATA = ["RP122_2025_OS_z08", "RP041_2025_OD_z10"]


def prepare_internal_data(
    input_folder: str,
    output_folder: str,
    label_key: str = "edit_v3",
    pixel_spacing: Tuple[float] = (3.87, 5.88),
):
    """Prepare internal data for nnU-Net training.
    This function reads data in H5 format and creates NIfTI files for training with nnU-Net.
    Input data has the format "RP<patient_id>_<year>_<eye_id>_z<slice_id>.h5", e.g. RP012_2023_OD_z08.h5.
    Output data has the format: oct_30<patient_id>_<year>_<eye_id>_z<slice_id>_0000.nii.gz,
    e.g. oct_30012_2023_OD_008_0000.nii.gz.

    Args:
        input_folder: Folder containing files in h5 format.
        output_folder: Output directory for imagesTr and labelsTr.
        label_key: Label key for path in h5 file.
        pixel_spacing: Voxel size for data.
    """
    # The affine matrix defines the spatial orientation and position
    # Default affine assumes the origin is at (0,0,0) and voxel spacing is as specified
    affine = np.eye(4)  # Identity matrix (standard for most cases)
    # µm per voxel (x, y, z)
    affine[0, 0] = pixel_spacing[0]  # x-spacing
    affine[1, 1] = pixel_spacing[1]  # y-spacing

    output_folder = os.path.realpath(output_folder)
    image_dir = os.path.join(output_folder, "imagesTr")
    label_dir = os.path.join(output_folder, "labelsTr")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    dataset_id = "30"

    files = glob(os.path.join(input_folder, "*.h5"))
    files.sort()
    for ff in tqdm(files, desc="Process files"):

        base_name = os.path.basename(ff).split(".h5")[0]

        # filter out datasets with missing layers
        if base_name in FILTER_DATA:
            print(f"Skipping {base_name}.")
            continue

        name_content = base_name.split("_")
        patient_id = name_content[0][2:]
        meas_year = name_content[1]
        eye_id = name_content[2]
        slice_id = name_content[3][1:]

        image = np.array(h5py.File(ff, "r")["image"])
        if label_key not in list(h5py.File(ff, "r")["labels"].keys()):
            print(f"Skipping {ff}. Label key {label_key} does not exist.")
            continue
        label = np.array(h5py.File(ff, "r")["labels"][label_key])

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)

        nnunet_identifier = f"{dataset_id}{patient_id.zfill(3)}_{meas_year}_{eye_id}_{slice_id.zfill(3)}"
        image_path = os.path.join(image_dir, f"oct_{nnunet_identifier}_0000.nii.gz")

        # Create the NIfTI image
        nifti_image = nib.Nifti1Image(image, affine)
        nib.save(nifti_image, image_path)

        nifti_label = nib.Nifti1Image(label, affine)
        label_path = os.path.join(label_dir, f"oct_{nnunet_identifier}.nii.gz")
        nib.save(nifti_label, label_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process internal training data for nnU-Net. Output data is stored in NIfTI format."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory with files in h5 format.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for creating imagesTr and labelsTr folders.")
    parser.add_argument("-l", "--label_key", type=str, default="edit_v1",
                        help="Description for data path to label data in h5 file. Default: 'edit_v3'.")

    args = parser.parse_args()

    prepare_internal_data(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()
