import os
from functools import partial
from typing import List, Optional

import h5py
import imageio.v3 as imageio
import napari
import numpy as np

from micro_sam.sam_annotator import image_series_annotator
from micro_sam.instance_segmentation import get_predictor_and_decoder
from micro_sam.util import precompute_image_embeddings
from qtpy.QtWidgets import QPushButton
from torch_em.util.segmentation import watershed_from_center_and_boundary_distances
from tqdm import tqdm

try:
    from micro_sam.instance_segmentation import get_amg
except ImportError:
    from micro_sam.instance_segmentation import get_instance_segmentation_generator as get_amg

from oct_tools.layer_information import get_layer_colormap
from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.napari_widgets.table_widget import MeasurementTableWidget
from oct_tools.napari_widgets.linelength_widget import LineLengthTableWidget
from oct_tools.napari_widgets.utils import _find_call_button, _measure, save_measurements


def _precompute_segmentation(images, sam_model_path, output_folder, postprocess=True,
                             postprocess_functions=["merge_horizontal", "filter_thin"],
                             use_prompts=True):
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

        foreground = segmenter._foreground
        boundary_distances = segmenter._boundary_distances
        center_distances = segmenter._center_distances

        if use_prompts:
            prompts = _derive_prompts_sam(foreground, boundary_distances)
            seg = _segment_from_prompts(predictor, image, prompts, min_size=150, embedding_path=embedding_path)
        else:
            seg = watershed_from_center_and_boundary_distances(center_distances, boundary_distances, foreground)

        if postprocess:
            seg = postprocess_segmentation(seg, image, verbose=False, postprocess_functions=postprocess_functions)

        imageio.imwrite(output_path, seg)

    return embedding_folder


def run_annotator(
    input_path: str,
    output_folder: str,
    slices: List[int],
    checkpoint_path: str,
    use_prompts: bool = True,
    precompute_segmentation: bool = True,
    postprocess_functions: List[str] = ["merge_horizontal", "filter_thin"],
    ref_position: Optional[int] = None,
    more_info: bool = False,
    color_style: str = "default",
):
    """Run annotator for a single or multiple slices of input data.
    A pre-computed segmentation can be used as an initial starting point.

    Args:
        input_path: Image data in TIF or H5 format.
        output_folder: Output folder for pre-computed segmentation.
        slices: Single or multiple slices of TIF data.
        checkpoint_path: File path to OCT-SAM model.
        use_prompts: Use two-phase prediction with prompts derived from first prediction.
        precompute_segmentation: Pre-compute SAM segmentation using SAM prompts.
        postprocessing_functions: List of functions. Post-processing will be performed in the given order.
        ref_position: Horizontal pixel coordinate of initial reference point for calculating layer thicknesses.
        more_info: Add additional information about layer length, max, min, and mean thickness.
        color_style: Color style for the visualization in napari.
    """
    basename = os.path.splitext(os.path.basename(input_path))[0]
    if ".h5" in input_path:
        images = [np.array(h5py.File(input_path, "r")["image"])]

    else:
        image_vol = imageio.imread(input_path)
        if len(image_vol.shape) == 3:
            images = [image_vol[z] for z in slices]
        elif len(image_vol.shape) == 2:
            images = [image_vol]
        else:
            raise ValueError("Check dimensionality of input TIF. Must be either 2D or 3D.")

    if precompute_segmentation:
        embedding_path = _precompute_segmentation(
            images, checkpoint_path, output_folder,
            postprocess_functions=postprocess_functions, use_prompts=use_prompts,
        )
    else:
        embedding_path = None

    # TODO use option for loading precomputed segmentations from another folder once this is in micro-sam master.
    viewer = image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=checkpoint_path,
        skip_segmented=False, return_viewer=True, embedding_path=embedding_path,
    )
    colormap = get_layer_colormap(color_style)
    if colormap is not None:
        viewer.layers["committed_objects"].colormap = colormap
        # add new committed objects layer
        new_arr = np.zeros(images[0].shape, dtype=np.uint8)
        viewer.add_labels(new_arr, visible=True, name="new_committed_objects")
        viewer.layers["new_committed_objects"].colormap = colormap

    # Add a button to trigger measurement saving
    save_func = partial(
        save_measurements,
        viewer=viewer,
        reference_name=basename,
        output_folder=output_folder,
        segmentation_layer_name="committed_objects",
        more_info=more_info
    )
    save_button = QPushButton("Save Measurements")
    save_button.clicked.connect(save_func)
    viewer.window.add_dock_widget(save_button, name="Save Measurements", area="bottom")

    # Get the next image button and bind the measurement function to it.
    next_image_button = _find_call_button(viewer, "Next Image [N]")
    next_image_button.clicked.connect(save_func)

    # widget measuring line lengths
    viewer.add_shapes(
        name="Lines",
        shape_type="line",
        edge_color="red",
        edge_width=2,
    )
    widget = LineLengthTableWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Line Length Measurer", area="right")

    # widget measuring fovea thickness and areas and layer thicknesses for reference point
    central_point = (images[0].shape[0] // 2, images[0].shape[1] // 2)
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
    viewer.window.add_dock_widget(measurement_widget, name="Measurement Table", area="right")

    napari.run()
