import argparse
import os
from glob import glob
from typing import List, Optional

import h5py
import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

from micro_sam.sam_annotator import image_series_annotator
from micro_sam.instance_segmentation import get_predictor_and_decoder
from micro_sam.util import precompute_image_embeddings
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QDockWidget, QPushButton
from tqdm import tqdm

try:
    from micro_sam.instance_segmentation import get_amg
except ImportError:
    from micro_sam.instance_segmentation import get_instance_segmentation_generator as get_amg

from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.segmentation_utils import run_measurement, get_etdrs_mask
from oct_tools.layer_information import identify_layers
from oct_tools.table_widget import MeasurementTableWidget


def _precompute_segmentation(images, sam_model_path, output_folder, postprocess=True,
                             postprocess_functions=["merge_horizontal", "filter_thin"]):
    """Precompute segmentation using SAM.
    """
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=sam_model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)
    os.makedirs(output_folder, exist_ok=True)

    # Precompute the embeddings.
    embedding_folder = os.path.join(output_folder, "embeddings")
    for i, image in tqdm(enumerate(images), desc="Precompute segmentation", total=len(images)):
        output_path = os.path.join(output_folder, f"seg_{i:05}.tif")
        if os.path.exists(output_path):
            continue

        # Init the segmenter for this image.
        embedding_path = os.path.join(embedding_folder, f"embedding_{i:05}.zarr")
        image_embeddings = precompute_image_embeddings(predictor, image, embedding_path, verbose=False)
        segmenter.initialize(image, verbose=False, image_embeddings=image_embeddings)
        foreground, boundary_distances = segmenter._foreground, segmenter._boundary_distances

        prompts = _derive_prompts_sam(foreground, boundary_distances)
        seg = _segment_from_prompts(predictor, image, prompts, min_size=150, embedding_path=embedding_path)
        if postprocess:
            seg = postprocess_segmentation(seg, image, verbose=False, postprocess_functions=postprocess_functions)

        imageio.imwrite(output_path, seg)

    return embedding_folder


def _find_call_button(viewer, button_text):
    for dw in viewer.window._qt_window.findChildren(QDockWidget):
        root = dw.widget()
        if root is None:
            continue
        for b in root.findChildren(QPushButton):
            if b.text() == button_text:
                return b
    raise RuntimeError(f"Could not find a QPushButton with text={button_text!r}")


def _measure(segmentation, fovea_point=None, reference_point=None, extra_information=False):
    n_layers = len(np.unique(segmentation)) - 1
    layer_mapping = identify_layers(segmentation, expected_number_of_layers=n_layers)
    layer_mapping = pd.DataFrame(dict(label_id=layer_mapping.keys(), layer=layer_mapping.values()))
    measurements = run_measurement(
        segmentation, extra_columns=layer_mapping, fovea_point=fovea_point, reference_point=reference_point,
        extra_information=extra_information,
    )
    etdrs_mask, notification_str = get_etdrs_mask(segmentation, measurements, fovea_point=fovea_point)

    # Reorder the columns so that the layer name is the second column.
    cols = measurements.columns.values.tolist()
    new_col_order = cols[:1] + cols[-1:] + cols[1:-1]
    measurements = measurements[new_col_order]
    return measurements, etdrs_mask, notification_str


def run_annotator(
    input_path: str,
    output_folder: str,
    slices: List[int],
    sam_model: str,
    precompute_segmentation: bool,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
    ref_position: Optional[int] = None,
    more_info: bool = False,
):
    """Run annotator for a single or multiple slices of input data.
    A pre-computed segmentation can be used as an initial starting point.

    Args:
        input_path: Image data in TIF or H5 format.
        output_folder: Output folder for pre-computed segmentation.
        slices: Single or multiple slices of TIF data.
        sam_model: File path to SAM model.
        precompute_segmentation: Pre-compute SAM segmentation using SAM prompts.
        postprocessing_functions: List of functions. Post-processing will be performed in the given order.
        ref_position: Horizontal pixel coordinate of initial reference point for calculating layer thicknesses.
        more_info: Add additional information about layer length, max, min, and mean thickness.
    """
    if ".h5" in input_path:
        images = [np.array(h5py.File(input_path, "r")["image"])]

    else:
        image_vol = imageio.imread(input_path)
        if len(image_vol.shape) == 3:
            images = [image_vol[z] for z in slices]
        elif len(image_vol.shape) == 2:
            images = [image_vol]
        else:
            raise ValueError("Check dimensionality of input TIF. Must be either 3D or 2D.")

    if precompute_segmentation:
        embedding_path = _precompute_segmentation(images, sam_model, output_folder,
                                                  postprocess_functions=postprocess_functions)
    else:
        embedding_path = None

    # TODO use option for loading precomputed segmentations from another folder once this is in micro-sam master.
    viewer = image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=sam_model,
        skip_segmented=False, return_viewer=True, embedding_path=embedding_path,
    )

    def post_measurement():
        segmentation = viewer.layers["committed_objects"].data
        measurements, _, _ = _measure(segmentation)
        i = len(glob(os.path.join(output_folder, "measurement*.tsv")))
        table_out = os.path.join(output_folder, f"measurement_{i:05}.tsv")
        measurements.to_csv(table_out, sep="\t", index=False)
        show_info(f"Measurements for saved to {table_out}")

    # Get the next image button and bind the measurement function to it.
    next_image_button = _find_call_button(viewer, "Next Image [N]")
    next_image_button.clicked.connect(post_measurement)

    central_point = (images[0].shape[0] // 2, images[0].shape[1] // 2)

    # set reference point for thickness measurement
    if ref_position is None:
        ref_point = (images[0].shape[0] // 2, images[0].shape[1] // 3)
    else:
        if ref_position >= images[0].shape[1] or ref_position < 0:
            ref_position = max([ref_position, 0])
            print(f"Position {ref_position} is outside the field of view and is adjusted.")
            ref_position = min([images[0].shape[1] - 1, ref_position])
        ref_point = (images[0].shape[0] // 2, ref_position)

    viewer.add_points(central_point, visible=True, name="fovea reference point", face_color="white")
    viewer.add_points(ref_point, visible=True, name="thickness reference point", face_color="blue")
    measurement_widget = MeasurementTableWidget(viewer, _measure, more_info)
    viewer.window.add_dock_widget(measurement_widget)
    napari.run()


def main():
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
        precompute_segmentation=args.precompute_segmentation,
        postprocess_functions=args.postprocess_functions,
        ref_position=args.ref_position,
        more_info=args.more_info,
    )


if __name__ == "__main__":
    main()
