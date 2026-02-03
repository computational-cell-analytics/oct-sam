import argparse
import os
from typing import List, Tuple, Union

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


def _load_h5_data(file_path, image_key):
    if isinstance(image_key, list):
        data = h5py.File(file_path, "r")[image_key[0]]
        for k in image_key[1:]:
            data = data[k]
        data = np.array(data)
    else:
        data = np.array(h5py.File(file_path, "r")[image_key])
    return data


def convert_data_to_nnunet(
    input_folder: str,
    output_folder: str,
    image_key: Union[str, List[str]] = "image",
    pixel_spacing: Tuple[float] = (3.87, 5.88),
):
    h5_paths = [entry.path for entry in os.scandir(input_folder) if ".h5" in entry.name]
    h5_paths.sort()

    # The affine matrix defines the spatial orientation and position
    # Default affine assumes the origin is at (0,0,0) and voxel spacing is as specified
    affine = np.eye(4)  # Identity matrix (standard for most cases)
    # µm per voxel (x, y, z)
    affine[0, 0] = pixel_spacing[0]  # x-spacing
    affine[1, 1] = pixel_spacing[1]  # y-spacing

    os.makedirs(output_folder, exist_ok=True)

    # get index for output in nnUNet format
    dataset_id = "01"

    for ff in tqdm(h5_paths, desc="Process files"):
        data = _load_h5_data(ff, image_key=image_key)

        base_name = os.path.basename(ff).split(".h5")[0]
        scan_id = base_name.split("_")[0][2:]

        slice_index = base_name[-2:]

        nnunet_identifier = f"{dataset_id}{str(scan_id).zfill(3)}{str(slice_index).zfill(3)}"
        image_path = os.path.join(output_folder, f"oct_{nnunet_identifier}_0000.nii.gz")
        nifti_image = nib.Nifti1Image(data, affine)
        nib.save(nifti_image, image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5 input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    convert_data_to_nnunet(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
    )


if __name__ == "__main__":
    main()
