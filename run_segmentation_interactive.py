import argparse
import os
from glob import glob
from typing import List

import h5py
import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

from micro_sam.sam_annotator import image_series_annotator
from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from micro_sam.util import precompute_image_embeddings
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QDockWidget, QPushButton
from tqdm import tqdm

from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.segmentation_utils import run_measurement
from oct_tools.layer_information import identify_layers
from oct_tools.table_widget import MeasurementTableWidget


def _precompute_segmentation(images, sam_model_path, output_folder, postprocess=True):
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
            seg = postprocess_segmentation(seg, image, verbose=False)

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


def _measure(segmentation):
    n_layers = len(np.unique(segmentation)) - 1
    layer_mapping = identify_layers(segmentation, expected_number_of_layers=n_layers)
    layer_mapping = pd.DataFrame(dict(label_id=layer_mapping.keys(), layer=layer_mapping.values()))
    measurements = run_measurement(segmentation, extra_columns=layer_mapping)
    # Reorder the columns so that the layer name is the second column.
    cols = measurements.columns.values.tolist()
    new_col_order = cols[:1] + cols[-1:] + cols[1:-1]
    measurements = measurements[new_col_order]
    return measurements


def run_annotator(
    input_path: str,
    output_folder: str,
    slices: List[int],
    sam_model: str,
    precompute_segmentation: bool,
    postprocess: bool = True,
):
    """Run annotator for a single or multiple slices of input data.
    A pre-computed segmentation can be used as an initial starting point.

    Args:
        input_path: Image data in TIF or H5 format.
        output_folder: Output folder for pre-computed segmentation.
        slices: Single or multiple slices of TIF data.
        sam_model: File path to SAM model.
        precompute_segmentation: Pre-compute SAM segmentation using SAM prompts.
        postprocess: Optional post-processing, e.g. removing thin lines, filling gaps in segmentation.
    """
    if ".h5" in input_path:
        images = [np.array(h5py.File(input_path, "r")["image"])]

    else:
        image_vol = imageio.imread(input_path)
        images = [image_vol[z] for z in slices]

    if precompute_segmentation:
        embedding_path = _precompute_segmentation(images, sam_model, output_folder, postprocess=postprocess)
    else:
        embedding_path = None

    # TODO use option for loading precomputed segmentations from another folder once this is in micro-sam master.
    viewer = image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=sam_model,
        skip_segmented=False, return_viewer=True, embedding_path=embedding_path,
    )

    def post_measurement():
        segmentation = viewer.layers["committed_objects"].data
        measurements = _measure(segmentation)
        i = len(glob(os.path.join(output_folder, "measurement*.tsv")))
        table_out = os.path.join(output_folder, f"measurement_{i:05}.tsv")
        measurements.to_csv(table_out, sep="\t", index=False)
        show_info(f"Measurements for saved to {table_out}")

    # Get the next image button and bind the measurement function to it.
    next_image_button = _find_call_button(viewer, "Next Image [N]")
    next_image_button.clicked.connect(post_measurement)

    measurement_widget = MeasurementTableWidget(viewer, _measure)
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

    args = parser.parse_args()
    run_annotator(
        args.input, args.output, args.slices, args.model, args.precompute_segmentation,
    )


if __name__ == "__main__":
    main()
