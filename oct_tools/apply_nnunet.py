import os
import subprocess
import tempfile
from collections import Counter
from typing import List, Optional, Tuple, Union

import h5py
import imageio.v3 as imageio
import nibabel as nib
import numpy as np
from tqdm import tqdm


def _load_h5_data(file_path: str, image_key: Union[str, List[str]]) -> np.ndarray:
    if isinstance(image_key, list):
        data = h5py.File(file_path, "r")[image_key[0]]
        for k in image_key[1:]:
            data = data[k]
        return np.array(data)
    return np.array(h5py.File(file_path, "r")[image_key])


def _convert_to_nnunet_format(
    input_folder: str,
    output_folder: str,
    image_key: Union[str, List[str]] = "image",
    pixel_spacing: Tuple[float, float] = (3.87, 5.88),
    file_format: Optional[str] = None,
) -> None:
    if file_format is None:
        file_formats = [
            os.path.splitext(e.name)[1][1:]
            for e in os.scandir(input_folder)
            if len(e.name.split(".")) > 1
        ]
        if not file_formats:
            raise ValueError(f"No eligible files in {input_folder}.")
        file_format = Counter(file_formats).most_common(1)[0][0]
        print(f"Automatically determined file format: {file_format}")

    file_paths = sorted(e.path for e in os.scandir(input_folder) if f".{file_format}" in e.name)

    affine = np.eye(4)
    affine[0, 0] = pixel_spacing[0]
    affine[1, 1] = pixel_spacing[1]

    os.makedirs(output_folder, exist_ok=True)
    print(f"Converting {len(file_paths)} files from {file_format} to NIfTI.")

    for ff in tqdm(file_paths, desc="Process files"):
        if file_format in ("h5", "H5"):
            data = _load_h5_data(ff, image_key=image_key)
        elif file_format.lower() in ("tif", "tiff"):
            data = imageio.imread(ff)
        else:
            raise ValueError(f"Unsupported file format: {file_format}.")

        base_name = os.path.splitext(os.path.basename(ff))[0]
        data = data.astype(np.uint8)

        if data.ndim == 3:
            for slice_id in range(data.shape[0]):
                out_path = os.path.join(output_folder, f"{base_name}_z{str(slice_id).zfill(3)}_0000.nii.gz")
                nib.save(nib.Nifti1Image(data[slice_id], affine), out_path)
        else:
            out_path = os.path.join(output_folder, f"{base_name}_0000.nii.gz")
            nib.save(nib.Nifti1Image(data, affine), out_path)


def _convert_nifti_to_tif(
    input_folder: str,
    output_folder: str,
    label_data: bool = True,
) -> None:
    file_paths = sorted(e.path for e in os.scandir(input_folder) if ".nii.gz" in e.name)
    os.makedirs(output_folder, exist_ok=True)

    for ff in file_paths:
        base_name = os.path.basename(ff).split(".nii.gz")[0]
        arr = nib.load(ff).get_fdata()
        if label_data:
            arr = arr.astype(np.uint32)
        imageio.imwrite(os.path.join(output_folder, f"{base_name}.tif"), arr)


def apply_model_nnunet(
    input_dir: str,
    output_dir: str,
    env_manager: str = "micromamba",
    env_nnunet: str = "nnunet",
    dataset_id: str = "001",
    configuration: str = "2d",
    fold: Union[int, str] = 0,
    device: str = "cpu",
) -> None:
    """Apply nnU-Net on all images in input_dir and write TIF segmentations to output_dir.

    Converts input images to NIfTI (nnU-Net format), runs nnUNetv2_predict via
    micromamba in the nnunet environment, then converts predictions back to TIF.

    Args:
        input_dir: Directory containing input images in TIF or H5 format.
        output_dir: Directory for TIF segmentation outputs.
        env_nnunet: micromamba environment name with nnU-Net installed.
        dataset_id: nnU-Net dataset ID, zero-padded (e.g. "001").
        configuration: nnU-Net configuration (e.g. "2d", "3d_fullres").
        fold: Fold index for prediction.
        device: Compute device ("cpu", "cuda", "mps").
    """
    with tempfile.TemporaryDirectory() as workdir:
        nifti_images = os.path.join(workdir, "images")
        nifti_segmentations = os.path.join(workdir, "segmentations")
        os.makedirs(nifti_images)
        os.makedirs(nifti_segmentations)

        _convert_to_nnunet_format(input_dir, nifti_images)

        subprocess.run(
            [
                env_manager, "run", "-n", env_nnunet,
                "nnUNetv2_predict",
                "-i", nifti_images,
                "-o", nifti_segmentations,
                "-d", dataset_id,
                "-c", configuration,
                "-f", str(fold),
                "-device", device,
            ],
            check=True,
        )

        _convert_nifti_to_tif(nifti_segmentations, output_dir)

    print("Output folder:", output_dir)
