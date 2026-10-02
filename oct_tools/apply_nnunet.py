import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from glob import glob
from typing import List, Optional, Tuple, Union

import h5py
import imageio.v3 as imageio
import nibabel as nib
import numpy as np
import torch
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
        # An RGB TIF holds a single B-scan, e.g. mouse OCT exported with three identical channels.
        if data.ndim == 3 and data.shape[-1] == 3:
            data = np.mean(data, axis=-1)
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
        if data.ndim == 3:
            for slice_id in range(data.shape[0]):
                out_path = os.path.join(output_folder, f"{base_name}_z{str(slice_id).zfill(3)}{suffix}.nii.gz")
                nib.save(nib.Nifti1Image(data[slice_id], affine), out_path)
        else:
            out_path = os.path.join(output_folder, f"{base_name}{suffix}.nii.gz")
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
        # get_fdata() always returns float64; dataobj keeps the stored dtype, e.g. uint8 for nnU-Net labels.
        arr = np.asanyarray(nib.load(ff).dataobj)
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
    trainer: str = "nnUNetTrainer",
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
        trainer: nnU-Net trainer class of the model, e.g. "nnUNetTrainer_250epochs".
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
                "-tr", trainer,
                "-device", device,
            ],
            check=True,
        )

        _convert_nifti_to_tif(nifti_segmentations, output_dir)

    print("Output folder:", output_dir)


def _create_dataset(image_dir: str, label_dir: str, dataset_dir: str, pretrained_json: dict, label_key: str) -> int:
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    _convert_to_nnunet_format(image_dir, images_tr)
    _convert_to_nnunet_format(
        label_dir, labels_tr,
        image_key=["labels", label_key],
        label_ids=sorted(pretrained_json["labels"].values()),
    )

    cases = {f[:-len(".nii.gz")] for f in os.listdir(labels_tr)}
    images = {f[:-len("_0000.nii.gz")]: f for f in os.listdir(images_tr)}
    orphans = cases - images.keys()
    if orphans:
        raise ValueError(f"Labels without an image in {image_dir}: {sorted(orphans)}")
    # The annotation can stop at any B-scan. Only the B-scans with a corrected label are training data.
    for case in images.keys() - cases:
        os.remove(os.path.join(images_tr, images[case]))

    dataset_json = {
        "channel_names": pretrained_json["channel_names"],
        "labels": pretrained_json["labels"],
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    with open(os.path.join(dataset_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent="\t")
    return len(cases)


def retrain_model_nnunet(
    input_dir: str,
    label_dir: str,
    pretrained_id: str,
    dataset_id: str,
    env_manager: str = "micromamba",
    env_nnunet: str = "nnunet",
    configuration: str = "2d",
    fold: Union[int, str] = 0,
    device: str = "cuda",
    trainer: str = "nnUNetTrainer",
    label_key: str = "edit_v3",
) -> None:
    """Fine-tune a trained nnU-Net model on images and corrected labels.

    Creates the nnU-Net dataset `dataset_id` in nnUNet_raw from all images that have a label,
    preprocesses it with the plans of the pretrained model, and runs nnUNetv2_train with the
    pretrained weights. The new model has the labels of the pretrained model. nnU-Net reads and
    writes the folders in the environment variables nnUNet_raw, nnUNet_preprocessed and nnUNet_results.
    If a folder of `dataset_id` exists, the function prints a warning with the folders and returns.

    Args:
        input_dir: Directory containing input images in TIF or H5 format.
        label_dir: Directory containing labels in TIF or H5 format. The label IDs must be IDs of
            the pretrained model. For a binary model, every value above 0 is foreground.
        pretrained_id: nnU-Net dataset ID of the pretrained model (e.g. "013").
        dataset_id: nnU-Net dataset ID of the new model. It must not exist.
        env_manager: Environment manager, e.g. "micromamba" or "conda".
        env_nnunet: Environment name with nnU-Net installed.
        configuration: nnU-Net configuration (e.g. "2d").
        fold: Fold index of the pretrained model and of the new model.
        device: Compute device ("cpu", "cuda", "mps").
        trainer: nnU-Net trainer class of the new model, e.g. "nnUNetTrainer_250epochs".
        label_key: Key of the label under `labels/` in H5 files.
    """
    env_vars = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    missing = [v for v in env_vars if v not in os.environ]
    if missing:
        raise RuntimeError(f"Set the environment variables {', '.join(missing)}. See doc/2d_mouse.md.")
    raw, preprocessed, results = (os.environ[v] for v in env_vars)

    prefix = f"Dataset{int(dataset_id):03d}_"
    existing = [path for root in (raw, preprocessed, results) for path in glob(os.path.join(root, prefix + "*"))]
    if existing:
        raise ValueError(
            f"Warning: The dataset ID {dataset_id} is in use, or a previous run with this ID was interrupted.\n"
            "Choose another ID, or delete these folders to use the ID again. "
            "A folder in nnUNet_results can contain a trained model.\n  " + "\n  ".join(existing)
        )

    # The trainer of the pretrained model can differ from the new trainer.
    pretrained = glob(os.path.join(
        results, f"Dataset{int(pretrained_id):03d}_*", f"*__nnUNetPlans__{configuration}"
    ))
    if len(pretrained) != 1:
        raise FileNotFoundError(
            f"Expected one '{configuration}' model of dataset {pretrained_id} in {results}, found {pretrained}."
        )
    checkpoint = os.path.join(pretrained[0], f"fold_{fold}", "checkpoint_final.pth")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Missing pretrained weights: {checkpoint}")
    # nnU-Net loads pretrained weights without map_location. A CUDA checkpoint then fails on the CPU or MPS.
    weights = torch.load(checkpoint, map_location="cpu", weights_only=False)["network_weights"]
    with open(os.path.join(pretrained[0], "dataset.json")) as f:
        pretrained_json = json.load(f)

    dataset_name = f"{prefix}OCT-2d-retrain-{int(pretrained_id):03d}"
    with tempfile.TemporaryDirectory() as workdir:
        dataset_dir = os.path.join(workdir, dataset_name)
        n_cases = _create_dataset(input_dir, label_dir, dataset_dir, pretrained_json, label_key)
        if n_cases < 5:
            raise ValueError(f"nnU-Net needs at least 5 labels for its 5-fold split, found {n_cases} in {label_dir}.")
        shutil.copytree(dataset_dir, os.path.join(raw, dataset_name))

    def run_nnunet(*args):
        subprocess.run([env_manager, "run", "-n", env_nnunet, *args], check=True)

    run_nnunet("nnUNetv2_plan_and_preprocess", "-d", dataset_id, "--verify_dataset_integrity", "--no_pp")

    # Pretrained weights load only into the same network, so the new dataset uses the pretrained plans.
    with open(os.path.join(pretrained[0], "plans.json")) as f:
        plans = json.load(f)
    plans["dataset_name"] = dataset_name
    with open(os.path.join(preprocessed, dataset_name, "nnUNetPlans.json"), "w") as f:
        json.dump(plans, f, indent="\t")

    run_nnunet("nnUNetv2_preprocess", "-d", dataset_id, "-c", configuration)
    pretrained_weights = os.path.join(preprocessed, dataset_name, "pretrained_weights.pth")
    torch.save({"network_weights": weights}, pretrained_weights)

    run_nnunet(
        "nnUNetv2_train", dataset_id, configuration, str(fold),
        "-tr", trainer,
        "-pretrained_weights", pretrained_weights,
        "-device", device,
    )
    print("Model folder:", os.path.join(results, dataset_name, f"{trainer}__nnUNetPlans__{configuration}"))
