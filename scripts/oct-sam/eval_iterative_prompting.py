import argparse
import os
from typing import List

from micro_sam.util import get_sam_model
from micro_sam.evaluation.inference import run_inference_with_iterative_prompting
from micro_sam.evaluation.evaluation import run_evaluation_for_iterative_prompting
from oct_tools.analysis.eval_iterative_prompting import create_individual_tif_data


def _run_iterative_prompting(
    output_folder: str,
    predictor,
    start_with_box_prompt: bool,
    image_paths: List[str],
    gt_paths: List[str],
    use_masks: bool = False,
):
    """Run iterative prompting for a selection of images and corresponding reference paths.

    Args:
        output_folder: Output directory for saving results and embeddings.
        predictor: SamPredictor - loaded model checkpoint.
        start_with_box_prompt: Flag for starting with box prompt.
        image_paths: Ordered list of image paths.
        gt_paths: Ordered list of label paths.
        use_masks:
    """
    prediction_root = os.path.join(
        output_folder, "start_with_box" if start_with_box_prompt else "start_with_point"
    )
    embedding_folder = os.path.join(output_folder, "embeddings")
    # image_paths, gt_paths = get_paths(dataset_name, split="test")
    run_inference_with_iterative_prompting(
        predictor=predictor,
        image_paths=image_paths,
        gt_paths=gt_paths,
        embedding_dir=embedding_folder,
        prediction_dir=prediction_root,
        start_with_box_prompt=start_with_box_prompt,
        use_masks=use_masks
    )
    return prediction_root


def _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, exp_folder, gt_paths):
    run_evaluation_for_iterative_prompting(
        gt_paths=gt_paths,
        prediction_root=prediction_root,
        experiment_folder=exp_folder,
        start_with_box_prompt=start_with_box_prompt,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate iterative prompting for SAM model."
    )
    parser.add_argument("-i", "--image_dir", type=str, required=True,
                        help="Directory containing image data in TIF format.")
    parser.add_argument("-l", "--label_dir", type=str, required=True,
                        help="Directory containing label data in TIF format.")

    parser.add_argument("-d", "--data_dir", type=str, default=None,
                        help="Directory featuring data in H5 format. Image_key: 'image'. Label_key: 'edit_v3'."
                        "Can be passed to create data for image and label directory.")

    parser.add_argument("-m", "--model", type=str, default="vit_b")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box.")

    args = parser.parse_args()

    if args.data_dir is not None:
        create_individual_tif_data(args.data_dir, args.image_dir, args.label_dir)

    start_with_box_prompt = args.box  # overwrite to start first iters' prompt with box instead of single point

    # get the predictor to perform inference
    predictor = get_sam_model(model_type=args.model, checkpoint_path=args.checkpoint)
    output_main = args.output_folder

    # label_ids to check
    label_ids = [i for i in range(1, 8)]

    for label in label_ids:
        image_tmp = os.path.join(args.image_dir, str(label))
        label_tmp = os.path.join(args.label_dir, str(label))
        output_folder = os.path.join(output_main, str(label))

        image_paths = [entry.path for entry in os.scandir(image_tmp) if ".tif" in entry.name]
        gt_paths = [entry.path for entry in os.scandir(label_tmp) if ".tif" in entry.name]
        image_paths.sort()
        gt_paths.sort()

        prediction_root = _run_iterative_prompting(
            output_folder=output_folder,
            predictor=predictor,
            start_with_box_prompt=start_with_box_prompt,
            image_paths=image_paths,
            gt_paths=gt_paths,
        )
        _evaluate_iterative_prompting(prediction_root, start_with_box_prompt, output_folder, gt_paths)


if __name__ == "__main__":
    main()
