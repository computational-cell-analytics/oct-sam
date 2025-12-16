import argparse
import os
from glob import glob
from typing import List

import h5py
import imageio.v3 as imageio
import napari
import numpy as np

from magicgui import magicgui
from micro_sam.sam_annotator import image_series_annotator
from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from tqdm import tqdm

from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.segmentation_utils import run_measurement


def _precompute_segmentation(images, sam_model_path, output_folder, postprocess=True):
    """Precompute segmentation using SAM.
    """
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=sam_model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)
    os.makedirs(output_folder, exist_ok=True)

    for i, image in tqdm(enumerate(images), desc="Precompute segmentation", total=len(images)):
        output_path = os.path.join(output_folder, f"seg_{i:05}.tif")

        # Init the segmenter for this image.
        segmenter.initialize(image, verbose=False)

        foreground, boundary_distances = segmenter._foreground, segmenter._boundary_distances

        prompts = _derive_prompts_sam(foreground, boundary_distances)
        seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
        if postprocess:
            seg = postprocess_segmentation(seg, image)

        imageio.imwrite(output_path, seg)


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
        _precompute_segmentation(images, sam_model, output_folder, postprocess=postprocess)

    # Another (better) option would be to just add this to the next button.
    # This should be relatively straightforward by getting the next button from the viewer
    # and adding this as an action.
    @magicgui(call_button="Run measurement")
    def measurement_widget(
        viewer: napari.Viewer,
        n_layers: int = 7,
    ):
        # TODO print info on where the result is saved and if the number of layers differs from the expectation.
        segmentation = viewer.layers["committed_objects"].data
        # TODO map the segmentation to layers and save it (via extra columns)
        measurements = run_measurement(segmentation)
        i = len(glob(os.path.join(output_folder, "measurement*.tsv")))
        table_out = os.path.join(output_folder, f"measurement_{i:05}.tsv")
        measurements.to_csv(table_out, sep="\t", index=False)

    # TODO use option for loading precomputed segmentations from another folder once this is in micro-sam master.
    viewer = image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=sam_model, skip_segmented=False, return_viewer=True,
    )
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
