import argparse
import os
from typing import List

from elf.evaluation import matching
from elf.evaluation.dice import symmetric_best_dice_score
from oct_tools.apply_oct_sam import apply_model_sam_2d, calc_accuracy
from oct_tools.data_reader import read_bscan


def eval_model_sam(
    input_path: str,
    checkpoint_path: str,
    save_folder: str,
    output_extension: str = "tif",
    force_overwrite: bool = False,
    view: bool = False,
    use_prompts: bool = True,
    postprocess: bool = False,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
    label_key: str = "original",
):
    """Evaluate an OCT-SAM model on 2D data.
    1) Apply an OCT-SAM model on the data.
    2) Optional: Evaluate the segmentation with respect to input labels. (only for H5 data)
    3) Optional: Visualize results with Napari.
    """
    # 1 - Apply OCT-SAM model on 2D slice data
    apply_model_sam_2d(input_path, checkpoint_path=checkpoint_path, save_folder=save_folder,
                       output_extension=output_extension,
                       postprocess_functions=postprocess_functions if postprocess else ["none"],
                       force_overwrite=force_overwrite,
                       use_prompts=use_prompts)

    # gather segmentation results
    if os.path.isdir(input_path):
        h5_paths = sorted(entry.path for entry in os.scandir(input_path) if entry.name.lower().endswith(".h5"))
    elif input_path.lower().endswith(".h5"):
        h5_paths = [input_path]
    else:
        h5_paths = []
    if not h5_paths:
        print("Evaluation needs labels from H5 input files. No H5 input found.")
        return

    names = [os.path.splitext(os.path.basename(h5_path))[0] for h5_path in h5_paths]
    seg_paths = [os.path.join(save_folder, f"{name}.{output_extension}") for name in names]
    segmentations = [read_bscan(seg_path, keys=("segmentation",)) for seg_path in seg_paths]

    # 2 - Evaluate segmentation accuracy
    images = [read_bscan(p) for p in h5_paths]
    labels = [read_bscan(p, keys=(f"labels/{label_key}",)) for p in h5_paths]
    calc_accuracy(labels, segmentations, names)

    # 3 - Visualize results with Napari
    if view:
        import napari
        for (image, label, seg, name) in zip(images, labels, segmentations, names):
            v = napari.Viewer()
            metrics = matching(seg, label)
            symm_dice = symmetric_best_dice_score(seg, label)

            msg = f"Image {name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
            msg += f", DICE={symm_dice:.3f}"
            v.title = msg
            v.add_image(image)
            v.add_labels(label)
            v.add_labels(seg)
            napari.run()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM model on all images in a folder. Evaluates data in H5 and/or TIF format."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory, which contains files in H5 or TIF format, or a single file.")
    parser.add_argument("-m", "--model", type=str, required=True)

    # arguments for saving output
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--output_extension", type=str, default="tif",
                        help="File extension for output. Either 'tif' or 'h5'. Default: tif")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # arguments for postprocessing
    parser.add_argument("--no_prompts", action="store_true",
                        help="Do not use two-phase prediction with prompts but only single prediction.")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--postprocess_functions", nargs="+", type=str,
                        default=["merge_horizontal", "filter_thin"],
                        help="Select and order post-processing functions 'merge_horizontal', 'filter_thin',"
                        "and 'fill_gaps'.")

    # arguments for evaluation
    parser.add_argument("--label_key", type=str, default="original",
                        help="Key for labels stored in H5 format.")
    parser.add_argument("--view", action="store_true",
                        help="Visually check segmentation using Napari.")

    args = parser.parse_args()

    eval_model_sam(
        input_path=args.input,
        checkpoint_path=args.model,
        save_folder=args.output,
        output_extension=args.output_extension,
        force_overwrite=args.force,
        view=args.view,
        use_prompts=not args.no_prompts,
        postprocess=args.postprocess,
        postprocess_functions=args.postprocess_functions,
        label_key=args.label_key,
    )


if __name__ == "__main__":
    main()
