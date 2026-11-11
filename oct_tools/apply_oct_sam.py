import os
from typing import List, Optional

import h5py
import imageio.v3 as imageio
import numpy as np
from tqdm import tqdm

from elf.evaluation import matching
from elf.evaluation.dice import symmetric_best_dice_score
from micro_sam.instance_segmentation import get_predictor_and_decoder
from torch_em.util.segmentation import watershed_from_center_and_boundary_distances

# To be compatible with the new and old micro-sam version.
try:
    from micro_sam.instance_segmentation import get_amg
except ImportError:
    from micro_sam.instance_segmentation import get_instance_segmentation_generator as get_amg

from oct_tools.data_reader import read_volume
from oct_tools.postprocessing import postprocess_segmentation, DEFAULT_POSTPROCESS
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts


def _segment_image(
    predictor,
    segmenter,
    image: np.ndarray,
    save_path: str,
    postprocess_functions: List[str] = DEFAULT_POSTPROCESS,
    use_prompts: bool = True,
):
    segmenter.initialize(image, verbose=False)
    foreground = segmenter._foreground
    boundary_distances = segmenter._boundary_distances
    center_distances = segmenter._center_distances

    print(f"Segmenting image: use_prompts {use_prompts}.")
    if use_prompts:
        prompts = _derive_prompts_sam(foreground, boundary_distances)
        seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
    else:
        prompts = None
        seg = watershed_from_center_and_boundary_distances(center_distances, boundary_distances, foreground)

    seg = postprocess_segmentation(seg, image, postprocess_functions)

    if save_path is not None:
        if save_path.endswith(".h5"):
            # Mode "a" keeps other datasets, e.g. "image" if the output is the input file.
            with h5py.File(save_path, "a") as f:
                for key in ("segmentation", "prompts"):
                    if key in f:
                        del f[key]
                f.create_dataset("segmentation", data=seg, compression="gzip")
                if prompts is not None:
                    f.create_dataset("prompts", data=prompts, compression="gzip")
        else:
            imageio.imwrite(save_path, seg)

    return seg, prompts


def _output_exists(save_path: str) -> bool:
    if not os.path.isfile(save_path):
        return False
    if save_path.endswith(".h5"):
        # The output H5 can be the input file, which exists before the segmentation.
        with h5py.File(save_path, "r") as f:
            return "segmentation" in f
    return True


def calc_accuracy(
    labels: List[np.ndarray],
    segmentations: List[np.ndarray],
    names: List[str],
):
    """Calculate accuracy of a segmentation compared to reference labels.
    The function prints the individual and overall values for precision,
    recall, f1-score, and the symmetric Dice's coefficient.

    Args:
        labels: List of labels, 2D numpy arrays.
        segmentations: List of segmentations, 2D numpy arrays.
        names: List of names for identification of results.
    """
    precisions, recalls, f1s = [], [], []
    symm_dice_scores = []
    for (label, seg, fname) in zip(labels, segmentations, names):
        metrics = matching(seg, label)
        symm_dice = symmetric_best_dice_score(seg, label)

        msg = f"Image {fname}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        msg += f", DICE={symm_dice:.3f}"
        print(msg)
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1s.append(metrics["f1"])
        symm_dice_scores.append(symm_dice)

    precision = np.round(np.mean(precisions), 3)
    recall = np.round(np.mean(recalls), 3)
    f1 = np.round(np.mean(f1s), 3)
    symm_dice_score = np.round(np.mean(symm_dice_scores), 3)

    print("Overall precision:", precision)
    print("Overall recall:", recall)
    print("Overall f1-score:", f1)
    print("Overall symm-dice:", symm_dice_score)


def apply_model_sam_2d(
    input_path: str,
    checkpoint_path: str,
    save_folder: str,
    output_extension: str = "tif",
    force_overwrite: bool = False,
    use_prompts: bool = True,
    postprocess_functions: List[str] = DEFAULT_POSTPROCESS,
    slices: Optional[List[int]] = None,
):
    """Apply OCT-SAM on 2D data.

    Args:
        input_path: Input path to data. Either a directory containing TIF, H5 or VOL files or a specific file path.
            A stack (z, y, x) gets one segmentation `<name>_<z>` for each B-scan.
        checkpoint_path: Path to the checkpoint from which to load the model.
        save_folder: Output folder for segmentation. With output_extension "h5", it can be the input folder.
            Then a TIF, VOL or stack input gets a new H5 file that holds only the segmentation. A later run reads
            this file as an input and stops with an error, because the file has no image.
        output_extension: File extension for segmentation.
        force_overwrite: Forcefully overwrite output.
        use_prompts: Use two-phase prediction with prompts derived from first prediction.
        postprocess_functions: List of sequential postprocessing functions.
        slices: The B-scans to segment in a stack. Default: all B-scans.
    """
    if os.path.isdir(input_path):
        file_paths = sorted(
            entry.path for entry in os.scandir(input_path)
            if entry.name.lower().endswith((".h5", ".tif", ".tiff", ".vol"))
        )
    elif os.path.isfile(input_path):
        file_paths = [os.path.realpath(input_path)]
    else:
        raise ValueError("Specify a viable path to an image or directory.")

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=checkpoint_path)
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)

    if use_prompts:
        print("Evaluating images with two-phase prediction using prompts derived from first prediction.")
    else:
        print("Evaluating images using single prediction.")

    input_paths = {os.path.realpath(path) for path in file_paths}
    for file_path in tqdm(file_paths, total=len(file_paths), desc="Segment images"):
        image = read_volume(file_path)
        basename = os.path.splitext(os.path.basename(file_path))[0]
        if len(image) > 1:
            z_ids = range(len(image)) if slices is None else slices
            invalid = [z for z in z_ids if not 0 <= z < len(image)]
            if invalid:
                raise ValueError(f"{file_path} has {len(image)} B-scans. Invalid slices: {invalid}.")
            bscans = [(image[z], f"{basename}_{z}") for z in z_ids]
        else:
            bscans = [(image[0], basename)]

        for bscan, name in bscans:
            save_path = os.path.join(save_folder, f"{name}.{output_extension}")
            target = os.path.realpath(save_path)
            # Only an H5 output may go into its own input file, because it keeps the image.
            if target in input_paths and (output_extension != "h5" or target != os.path.realpath(file_path)):
                raise ValueError(f"The output {save_path} would replace an input file. Choose another output folder.")
            if _output_exists(save_path) and not force_overwrite:
                print(f"Skipping {name}. Segmentation already exists.")
                continue
            _, _ = _segment_image(
                predictor, segmenter, bscan, save_path, postprocess_functions, use_prompts=use_prompts,
            )
