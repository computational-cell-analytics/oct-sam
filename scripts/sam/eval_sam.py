import argparse
import os

import h5py
import imageio.v3 as imageio
import napari
import numpy as np
from tqdm import tqdm

from elf.evaluation import matching
from elf.evaluation.dice import symmetric_best_dice_score
from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts


def _segment_image(predictor, segmenter, image, save_path, postprocess=False,
                   postprocess_functions=["merge_horizontal", "filter_thin"]):
    if save_path is not None and os.path.exists(save_path) and ".h5" in save_path:
        with h5py.File(save_path, "r") as f:
            return f["segmentation"][:], f["prompts"][:]

    segmenter.initialize(image)
    foreground, boundary_distances = segmenter._foreground, segmenter._boundary_distances

    prompts = _derive_prompts_sam(foreground, boundary_distances)
    seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
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


def eval_model_sam(input_dir, model_path, save_folder=None, view=False, postprocess=False,
                   postprocess_functions=["merge_horizontal", "filter_thin"]):
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)

    h5_paths = [entry.path for entry in os.scandir(input_dir) if ".h5" in entry.name]
    h5_paths.sort()
    images = [np.array(h5py.File(p, "r")["image"]) for p in h5_paths]
    labels = [np.array(h5py.File(p, "r")["labels"]["original"]) for p in h5_paths]

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    segmentations, prompts = [], []
    for h5_path, image in tqdm(zip(h5_paths, images), total=len(images), desc="Segment images"):
        basename = "".join(os.path.basename(h5_path).split(".")[:-1])
        save_path = None if save_folder is None else os.path.join(save_folder, f"{basename}.tif")
        seg, this_prompts = _segment_image(predictor, segmenter, image, save_path, postprocess, postprocess_functions)
        segmentations.append(seg)
        prompts.append(this_prompts)

    precisions, recalls, f1s = [], [], []
    symm_dice_scores = []
    for i, (image, label, seg) in enumerate(zip(images, labels, segmentations)):
        h5_path = h5_paths[i]
        fname = os.path.basename(h5_path)
        metrics = matching(seg, label)
        symm_dice = symmetric_best_dice_score(seg, label)

        msg = f"Image {fname}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
        msg += f", DICE={symm_dice:.3f}"
        if view:
            point_prompts = prompts[i]
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(label)
            v.add_labels(seg)
            v.add_points(point_prompts, visible=False)
            v.title = msg
            napari.run()
        else:
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM model on all images in a folder. Evaluates data in h5 format."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="./oct-sam-v3.pt")
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument("--output_extension", type=str, default="tif",
                        help="File extension for output. Either 'tif' or 'h5'. Default: tif")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--postprocess-functions", nargs="+", type=str,
                        default=["merge_horizontal", "filter_thin"],
                        help="Select and order post-processing functions 'merge_horizontal', 'filter_thin',"
                        "and 'fill_gaps'.")

    args = parser.parse_args()

    eval_model_sam(args.input_dir, args.model, args.output_dir, args.check,
                   args.postprocess, args.postprocess_functions)


if __name__ == "__main__":
    main()
