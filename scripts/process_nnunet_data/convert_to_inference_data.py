import argparse
import os
from collections import Counter
from typing import List, Optional, Tuple, Union

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
    file_format: Optional[str] = None,
):
    # find most common file format in input directory
    if file_format is None:
        file_formats = [os.path.splitext(entry.name)[1][1:] for entry in os.scandir(input_folder) if
                        len(entry.name.split(".")) > 1]
        if len(file_formats) == 0:
            raise ValueError(f"No elligible file format in input directory {input_folder}. Check data.")
        data = Counter(file_formats)
        file_format = data.most_common(1)[0][0]
        print(f"Automatically determined file format: {file_format}")

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
    print(f"Converting {len(file_paths)} files from {file_format} to NIfTI.")

    for ff in tqdm(file_paths, desc="Process files"):
        if file_format in ["h5", "H5"]:
            data = _load_h5_data(ff, image_key=image_key)
        elif file_format in ["TIF", "TIFF", "tif", "tiff"]:
            data = imageio.imread(ff)
        else:
            raise ValueError(f"Unsupported file format: {file_format}.")

        base_name = os.path.splitext(os.path.basename(ff))[0]
        data = data.astype(np.uint8)

        if len(data.shape) == 3:
            slice_number = data.shape[0]
            for slice_id in range(slice_number):
                data_slice = data[slice_id, :, :]
                image_path = os.path.join(output_folder, f"{base_name}_z{str(slice_id).zfill(3)}_0000.nii.gz")
                nifti_image = nib.Nifti1Image(data_slice, affine)
                nib.save(nifti_image, image_path)
        else:
            image_path = os.path.join(output_folder, f"{base_name}_0000.nii.gz")
            nifti_image = nib.Nifti1Image(data, affine)
            nib.save(nifti_image, image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Convert input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing image data.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for converted NIfTI files.")
    parser.add_argument("-f", "--file_format", type=str, default=None, choices=[None, "h5", "tif"],
                        help="File format of input data. Default: Most frequent file type in input directory.")

    args = parser.parse_args()

    convert_data_to_nnunet(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        file_format=args.file_format,
    )


if __name__ == "__main__":
    main()
