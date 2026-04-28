"""private
"""
import argparse

from oct_tools.interactive_segmentation import run_annotator
from oct_tools.metric_utils import calculate_metrics
from oct_tools.apply_oct_sam import apply_model_sam_2d


def interactive():
    parser = argparse.ArgumentParser(
        description="Apply SAM model on a single or multiple slices of input data."
    )
    parser.add_argument("-i", "--input", required=True, help="Input image.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("-z", "--slices", nargs="+", type=int, default=[0],
                        help="Slice(s) in z-direction. The first slice if taken by default.")
    parser.add_argument("--model", required=True, help="The path to the segmentation model.")
    parser.add_argument("--precompute_segmentation", action="store_true",
                        help="Pre-compute segmentation using prompts derived from SAM prediction.")
    parser.add_argument("--postprocess_functions", nargs="+", type=str,
                        default=["merge_horizontal", "filter_thin"],
                        help="Select and order post-processing functions 'merge_horizontal', 'filter_thin',"
                        "and 'fill_gaps'. Use 'no' or 'none' for no post-processing.")
    parser.add_argument("--no_prompts", action="store_true",
                        help="Do not use two-phase prediction with prompts but only single prediction.")
    parser.add_argument("--ref_position", type=int, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument(
        "--more_info", action="store_true",
        help="Display additional information (length, max_thickness, min_thickness, etc.) in measuremnt table.",
    )

    args = parser.parse_args()
    run_annotator(
        args.input, args.output,
        slices=args.slices,
        checkpoint_path=args.model,
        use_prompts=not args.no_prompts,
        precompute_segmentation=args.precompute_segmentation,
        postprocess_functions=args.postprocess_functions,
        ref_position=args.ref_position,
        more_info=args.more_info,
    )


def metrics():
    parser = argparse.ArgumentParser(
        description="Calculate OCT-metrics for 2D segmentation. "
        "The specific layer thickness can be calculated at a reference position on the horizontal axis. "
        "An ETDRS grid can be created for a given fovea position."
    )
    parser.add_argument("-i", "--input", type=str, required=True, help="Input segmentation.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path. Supports 'tsv' and 'xlsx' as file extensions.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+",
                        default=[3.87166976, 5.8814],
                        help="Voxel size of 2D input in micrometer.")
    parser.add_argument("--ref_position", type=float, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument("--fovea", type=float, default=None,
                        help="Position of fovea point on vertical axis for calculating area of ETDRS grid.")
    parser.add_argument("--etdrs_grid", type=str, default=None,
                        help="File path to export ETDRS grid.")

    args = parser.parse_args()

    calculate_metrics(
        args.input, args.output, args.voxel_size,
        fovea_position=args.fovea,
        reference_position=args.ref_position,
        etdrs_grid=args.etdrs_grid,
    )


def apply_sam():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM model on all images in a folder. Evaluates data in H5 format."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory, which contains files in H5 or TIF format, or a specific file path.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="The path to the segmentation model.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--output_extension", type=str, default="tif",
                        help="File extension for output. Either 'tif' or 'h5'. Default: tif")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--no_prompts", action="store_true",
                        help="Do not use two-phase prediction with prompts but only single prediction.")
    parser.add_argument("--postprocess_functions", nargs="+", type=str,
                        default=["merge_horizontal", "filter_thin"],
                        help="Select and order post-processing functions 'merge_horizontal', 'filter_thin',"
                        "and 'fill_gaps'. Use 'no' or 'none' for no post-processing.")

    args = parser.parse_args()

    apply_model_sam_2d(
        input_path=args.input,
        checkpoint_path=args.model,
        save_folder=args.output,
        output_extension=args.output_extension,
        force_overwrite=args.force,
        use_prompts=not args.no_prompts,
        postprocess_functions=args.postprocess_functions,
    )
