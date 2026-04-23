"""private
"""
import argparse

from oct_tools.interactive_segmentation import run_annotator


def interactive():
    parser = argparse.ArgumentParser(
        description="Apply SAM model on a single or multiple slices of input data."
    )
    parser.add_argument("-i", "--input", required=True, help="Input image.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("-z", "--slices", nargs="+", type=int, required=True, help="Slice(s) in z-direction.")
    parser.add_argument("--model", default="./oct-sam-v4.pt", help="The SAM model trained for OCT data model.")
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
        sam_model=args.model,
        use_prompts=not args.no_prompts,
        precompute_segmentation=args.precompute_segmentation,
        postprocess_functions=args.postprocess_functions,
        ref_position=args.ref_position,
        more_info=args.more_info,
    )
