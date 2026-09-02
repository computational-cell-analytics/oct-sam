"""private
"""
import argparse

import imageio.v3 as imageio
import napari
import numpy as np
from h5py import File

from oct_tools.interactive_segmentation import run_annotator
from oct_tools.layer_information import get_layer_colormap
from oct_tools.metric_utils import calculate_metrics
from oct_tools.apply_oct_sam import apply_model_sam_2d
from oct_tools.apply_nnunet import apply_model_nnunet
from oct_tools.eval_segmentation import eval_segmentation_2d
from oct_tools.measure_segmentation import run_measurement_only
from oct_tools.napari_widgets.colormap_widget import ColormapWidget


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
                        default=["merge_horizontal", "filter_thin", "assign_layer_id"],
                        help="Select and order post-processing functions: 'merge_horizontal', 'filter_thin',"
                        " 'fill_gaps', 'assign_layer_id'. Use 'no' or 'none' for no post-processing.")
    parser.add_argument("--no_prompts", action="store_true",
                        help="Do not use two-phase prediction with prompts but only single prediction.")
    parser.add_argument("--ref_position", type=int, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument(
        "--more_info", action="store_true",
        help="Display additional information (length, max_thickness, min_thickness, etc.) in measurement table.",
    )
    parser.add_argument(
        "--color_style", type=str, default="custom", choices=["default", "custom", "random", "check"],
        help="Label color scheme for napari: 'default', 'custom', or 'random'.",
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
        color_style=args.color_style,
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
                        help="Voxel size of the 2D input in micrometer, as (vertical, horizontal). "
                        "Vertical is across the retinal layers, horizontal is along them. "
                        "A single value is used for both axes.")
    parser.add_argument("--ref_position", type=float, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument("--fovea", type=float, default=None,
                        help="Position of foveal point on vertical axis for calculating area of ETDRS grid.")
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
                        default=["merge_horizontal", "filter_thin", "assign_layer_id"],
                        help="Select and order post-processing functions: 'merge_horizontal', 'filter_thin',"
                        " 'fill_gaps', 'assign_layer_id'. Use 'no' or 'none' for no post-processing.")

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


def apply_nnunet():
    parser = argparse.ArgumentParser(
        description="Apply nnU-Net on all images in a folder. Outputs segmentations in TIF format."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory containing images in TIF or H5 format.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output directory for TIF segmentations.")
    parser.add_argument("-m", "--env_manager", type=str, default="micromamba",
                        help="Environment manager, e.g. micromamba or conda. Default: micromamba")
    parser.add_argument("-n", "--env_nnunet", type=str, default="nnunet",
                        help="Environment name with nnU-Net installed. Default: nnunet")
    parser.add_argument("-d", "--dataset_id", type=str, default="001",
                        help="nnU-Net dataset ID (zero-padded). Default: 001")
    parser.add_argument("-c", "--configuration", type=str, default="2d",
                        help="nnU-Net configuration (e.g. '2d', '3d_fullres'). Default: 2d")
    parser.add_argument("-f", "--fold", type=int, default=0,
                        help="Fold index for nnU-Net prediction. Default: 0")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Compute device for nnU-Net. Default: cpu")

    args = parser.parse_args()
    apply_model_nnunet(
        input_dir=args.input,
        output_dir=args.output,
        env_manager=args.env_manager,
        env_nnunet=args.env_nnunet,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        fold=args.fold,
        device=args.device,
    )


def eval_segmentation():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation performance on all images in a folder. "
        "Evaluates image and label data in H5 format and segmentation data in TIF format."
    )
    parser.add_argument("-d", "--data_dir", type=str, required=True,
                        help="Directory containing images and labels in H5 format.")
    parser.add_argument("-s", "--seg_dir", type=str, required=True,
                        help="Directory containing segmentation in TIF format.")
    parser.add_argument("--nnunet", action="store_true",
                        help="Check for nnU-Net inference format.")
    parser.add_argument("--label_key", type=str, default="original",
                        help="Key for labels stored in H5 format.")
    parser.add_argument("--json", type=str, default=None,
                        help="Output path for JSON dictionary documenting performance.")

    args = parser.parse_args()

    eval_segmentation_2d(
        args.data_dir,
        args.seg_dir,
        check_nnunet=args.nnunet,
        label_key=args.label_key,
        json_file=args.json,
    )


def measure():
    parser = argparse.ArgumentParser(
        description="Measure segmentation metrics using napari."
    )
    parser.add_argument("-i", "--img", required=True, help="Image path.")
    parser.add_argument("-s", "--seg", required=True, help="Segmentation path.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("-z", "--slice", type=int, default=0,
                        help="Slice in z-direction. The first slice if taken by default.")
    parser.add_argument("--ref_position", type=int, default=None,
                        help="Initial position on vertical axis of reference point for calculating layer thickness.")
    parser.add_argument(
        "--more_info", action="store_true",
        help="Display additional information (length, max_thickness, min_thickness, etc.) in measuremnt table.",
    )
    parser.add_argument(
        "--color_style", type=str, default="custom", choices=["default", "custom", "random", "check"],
        help="Label color scheme for napari: 'default', 'custom', or 'random'.",
    )

    args = parser.parse_args()

    run_measurement_only(
        image_path=args.img,
        segmentation_path=args.seg,
        output_folder=args.output,
        ref_position=args.ref_position,
        more_info=args.more_info,
        slice_index=args.slice,
        color_style=args.color_style,
    )


def open_labels():
    parser = argparse.ArgumentParser(
        description="Open one or more segmentation files in napari with a custom label color map."
    )
    parser.add_argument(
        "files", nargs="+",
        help="Segmentation files in TIF or H5 format.",
    )
    parser.add_argument(
        "--color_style", type=str, default="custom", choices=["default", "custom", "random", "check"],
        help="Initial label color scheme: 'default', 'custom', or 'random'.",
    )
    parser.add_argument(
        "--slice", type=int, default=0,
        help="Slice index for 3D TIF files (default: 0).",
    )

    args = parser.parse_args()
    colormap = get_layer_colormap(args.color_style)

    viewer = napari.Viewer()

    for filepath in args.files:
        if filepath.endswith(".h5"):
            with File(filepath, "r") as f:
                for key in ("segmentation", "seg"):
                    if key in f:
                        data = f[key][:]
                        break
                else:
                    raise KeyError(f"{filepath}: no 'segmentation' or 'seg' dataset found.")
        elif filepath.endswith(".tif"):
            vol = imageio.imread(filepath)
            data = vol[args.slice] if vol.ndim == 3 else vol
        else:
            raise ValueError(f"Unsupported format for {filepath}. Use .tif or .h5.")

        if not np.issubdtype(data.dtype, np.integer):
            data = data.astype(np.uint32)

        layer = viewer.add_labels(data, name=filepath)
        if colormap is not None:
            layer.colormap = colormap

    widget = ColormapWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Label Color Map", area="right")

    napari.run()
