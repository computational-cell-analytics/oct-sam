import os
import shutil
import subprocess
import tempfile
from collections import Counter
from glob import glob
from typing import List, Optional, Sequence, Tuple, Union

import imageio.v3 as imageio
import nibabel as nib
import numpy as np
from tqdm import tqdm

from oct_tools.data.data_reader import IMAGE_KEYS, nifti_affine, read_nifti, read_volume


def _convert_to_nnunet_format(
    input_folder: str,
    output_folder: str,
    keys: Sequence[str] = IMAGE_KEYS,
    pixel_spacing: Tuple[float, float] = (3.87, 5.88),
    file_format: Optional[str] = None,
    label_ids: Optional[List[int]] = None,
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

    file_paths = sorted(e.path for e in os.scandir(input_folder) if e.name.endswith(f".{file_format}"))

    affine = nifti_affine(pixel_spacing)

    os.makedirs(output_folder, exist_ok=True)
    print(f"Converting {len(file_paths)} files from {file_format} to NIfTI.")

    for ff in tqdm(file_paths, desc="Process files"):
        data = read_volume(ff, keys)
        base_name = os.path.splitext(os.path.basename(ff))[0]
        # label_ids marks label data. It holds the label IDs of the model.
        if label_ids is not None:
            # A binary model takes every ID above 0 as foreground, e.g. a committed SAM object in annotate.py.
            if label_ids == [0, 1]:
                data = data > 0
            unexpected = sorted(set(np.unique(data).tolist()) - set(label_ids))
            if unexpected:
                raise ValueError(f"{ff} has the label IDs {unexpected}. The model has only the IDs {label_ids}.")
        data = data.astype(np.uint8)

        # nnU-Net needs the channel suffix on images and forbids it on labels.
        suffix = "_0000" if label_ids is None else ""
        if len(data) > 1:
            for slice_id in range(len(data)):
                out_path = os.path.join(output_folder, f"{base_name}_z{str(slice_id).zfill(3)}{suffix}.nii.gz")
                nib.save(nib.Nifti1Image(data[slice_id], affine), out_path)
        else:
            out_path = os.path.join(output_folder, f"{base_name}{suffix}.nii.gz")
            nib.save(nib.Nifti1Image(data[0], affine), out_path)


def _convert_nifti_to_tif(
    input_folder: str,
    output_folder: str,
    label_data: bool = True,
) -> None:
    file_paths = sorted(e.path for e in os.scandir(input_folder) if e.name.endswith(".nii.gz"))
    os.makedirs(output_folder, exist_ok=True)

    for ff in file_paths:
        base_name = os.path.basename(ff).split(".nii.gz")[0]
        arr = read_nifti(ff)
        if label_data:
            arr = arr.astype(np.uint32)
        imageio.imwrite(os.path.join(output_folder, f"{base_name}.tif"), arr)


def apply_model_nnunet(
    input_path: str,
    output_path: str,
    env_manager: str = "micromamba",
    env_nnunet: str = "nnunet",
    dataset_id: str = "001",
    configuration: str = "2d",
    fold: Union[int, str] = 0,
    device: str = "cpu",
    trainer: str = "nnUNetTrainer",
) -> None:
    """Apply nnU-Net on an image file or on all images in a directory and write TIF segmentations.

    Converts input images to NIfTI (nnU-Net format), runs nnUNetv2_predict via
    micromamba in the nnunet environment, then converts predictions back to TIF.

    Args:
        input_path: Image file, or directory of images, in TIF, H5 or VOL format.
        output_path: Segmentation TIF file for a file input. Output directory for a directory input.
            The TIF of a stack holds one segmentation for each B-scan.
        env_manager: Environment manager, e.g. "micromamba" or "conda".
        env_nnunet: Environment name with nnU-Net installed.
        dataset_id: nnU-Net dataset ID, zero-padded (e.g. "001").
        configuration: nnU-Net configuration (e.g. "2d", "3d_fullres").
        fold: Fold index for prediction.
        device: Compute device ("cpu", "cuda", "mps").
        trainer: nnU-Net trainer class of the model, e.g. "nnUNetTrainer_250epochs".
    """
    if os.path.realpath(output_path) == os.path.realpath(input_path):
        raise ValueError(f"The output {output_path} is the input. Choose another output path.")
    single_file = os.path.isfile(input_path)
    if single_file:
        extension = os.path.splitext(input_path)[1].lower()
        if extension not in (".tif", ".tiff", ".h5", ".vol"):
            raise ValueError(f"Unsupported input file {input_path}. Use a TIF, H5 or VOL file.")
        if not output_path.lower().endswith((".tif", ".tiff")):
            raise ValueError(f"The output for the input file {input_path} must be a TIF file, not {output_path}.")

    with tempfile.TemporaryDirectory() as workdir:
        nifti_images = os.path.join(workdir, "images")
        nifti_segmentations = os.path.join(workdir, "segmentations")
        os.makedirs(nifti_images)
        os.makedirs(nifti_segmentations)

        input_dir = input_path
        if single_file:
            # A fixed name makes the nnU-Net case name independent of the input file name.
            input_dir = os.path.join(workdir, "input")
            os.makedirs(input_dir)
            shutil.copy(input_path, os.path.join(input_dir, f"bscan{extension}"))

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
                "-tr", trainer,
                "-device", device,
            ],
            check=True,
        )

        if single_file:
            # A stack gives one prediction for each B-scan, sorted by the zero-padded z index.
            predictions = [read_nifti(path) for path in sorted(glob(os.path.join(nifti_segmentations, "*.nii.gz")))]
            segmentation = predictions[0] if len(predictions) == 1 else np.stack(predictions)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            imageio.imwrite(output_path, segmentation.astype(np.uint32))
        else:
            _convert_nifti_to_tif(nifti_segmentations, output_path)

    print("Output:", output_path)
