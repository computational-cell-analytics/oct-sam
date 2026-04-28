import argparse
import os
from typing import List, Tuple, Union

import h5py
import imageio.v3 as imageio
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
    file_format: str = "h5",
):
    file_paths = [entry.path for entry in os.scandir(input_folder) if f".{file_format}" in entry.name]
    file_paths.sort()

    # The affine matrix defines the spatial orientation and position
    # Default affine assumes the origin is at (0,0,0) and voxel spacing is as specified
    affine = np.eye(4)  # Identity matrix (standard for most cases)
    # µm per voxel (x, y, z)
    affine[0, 0] = pixel_spacing[0]  # x-spacing
    affine[1, 1] = pixel_spacing[1]  # y-spacing

    os.makedirs(output_folder, exist_ok=True)

    # get index for output in nnUNet format

    for ff in tqdm(file_paths, desc="Process files"):
        if file_format in ["h5", "H5"]:
            data = _load_h5_data(ff, image_key=image_key)
        elif file_format in ["TIF", "TIFF", "tif", "tiff"]:
            data = imageio.imread(ff)
        else:
            raise ValueError(f"Unsupported file format: {file_format}.")

        base_name = os.path.basename(ff).split(f".{file_format}")[0]

        name_content = base_name.split("_")
        patient_id = name_content[0]
        meas_year = name_content[1]
        eye_id = name_content[2]
        if len(name_content) == 4:
            slice_id = name_content[3][1:]
        elif len(name_content) != 3:
            raise ValueError(f"File {base_name} does not correspond to file format RP<id>_<year>_<eye_id> "
                             "or RP<id>_<year>_<eye_id>_z<slice_id>.")

        if "_z" in base_name:
            slice_id = base_name[-2:]
        else:
            slice_id = None

        data = data.astype(np.uint8)
        if slice_id is None:
            slice_number = data.shape[0]
            for slice_id in range(slice_number):
                data_slice = data[slice_id, :, :]
                nnunet_identifier = f"{patient_id}_{meas_year}_{eye_id}_z{str(slice_id).zfill(3)}"
                image_path = os.path.join(output_folder, f"oct_{nnunet_identifier}_0000.nii.gz")
                nifti_image = nib.Nifti1Image(data_slice, affine)
                nib.save(nifti_image, image_path)

        else:
            nnunet_identifier = f"{patient_id}_{meas_year}_{eye_id}_z{str(slice_id).zfill(3)}"
            image_path = os.path.join(output_folder, f"oct_{nnunet_identifier}_0000.nii.gz")
            nifti_image = nib.Nifti1Image(data, affine)
            nib.save(nifti_image, image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert H5 input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-f", "--file_format", type=str, default="h5",
                        help="File format of input data. Default: h5")

    args = parser.parse_args()

    convert_data_to_nnunet(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        file_format=args.file_format,
    )


if __name__ == "__main__":
    main()
