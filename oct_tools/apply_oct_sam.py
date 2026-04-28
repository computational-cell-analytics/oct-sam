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

from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts


def _segment_image(
    predictor,
    segmenter,
    image: np.ndarray,
    save_path: str,
    postprocess: bool = False,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
    use_prompts: bool = True,
):
    if save_path is not None and os.path.exists(save_path) and ".h5" in save_path:
        with h5py.File(save_path, "r") as f:
            return f["segmentation"][:], f["prompts"][:]

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

    if postprocess:
        seg = postprocess_segmentation(seg, image, postprocess_functions)

    if save_path is not None:
        if ".h5" in save_path:
            with h5py.File(save_path, "a") as f:
                f.create_dataset("segmentation", data=seg, compression="gzip")
                f.create_dataset("prompts", data=prompts, compression="gzip")
        else:
            imageio.imwrite(save_path, seg)

    return seg, prompts


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
    postprocess: bool = False,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
):
    """Apply OCT-SAM on 2D data.

    Args:
        input_path: Input path to data. Either a directory containing TIF and/or H5 files or a specific file path.
        checkpoint_path: Path to the checkpoint from which to load the model.
        save_folder: Output folder for segmentation.
        output_extension: File extension for segmentation.
        force_overwrite: Forcefully overwrite output.
        use_prompts: Use two-phase prediction with prompts derived from first prediction.
        postprocess: Flag for postprocessing segmentation.
        postprocess_functions: List of sequential postprocessing functions.
    """
    if os.path.isdir(input_path):
        # gather viable files in input directory
        viable_extensions = ["h5", "tif"]
        file_paths = [entry.path for entry in os.scandir(input_path) if
                      any([ext in entry.name for ext in viable_extensions])]
        file_paths.sort()
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

    for file_path in tqdm(file_paths, total=len(file_paths), desc="Segment images"):
        # read in data dependent on file extension
        if ".h5" in file_path:
            image = np.array(h5py.File(file_path, "r")["image"])
        elif ".tif" in file_path:
            image = imageio.imread(file_path)

        basename = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_folder, f"{basename}.{output_extension}")
        if os.path.isfile(save_path) and not force_overwrite:
            print(f"Skipping {basename}. Segmentation already exists.")
            continue
        _, _ = _segment_image(
            predictor, segmenter, image, save_path, postprocess, postprocess_functions, use_prompts=use_prompts,
        )
