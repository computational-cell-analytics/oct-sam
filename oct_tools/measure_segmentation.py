import os
from functools import partial
from typing import Optional

import numpy as np
import napari

from oct_tools.data_reader import SEGMENTATION_KEYS, read_bscan, read_volume
from oct_tools.layer_information import get_layer_colormap
from oct_tools.napari_widgets.colormap_widget import ColormapWidget
from oct_tools.napari_widgets.table_widget import MeasurementTableWidget
from oct_tools.napari_widgets.linelength_widget import LineLengthTableWidget
from oct_tools.napari_widgets.utils import _measure, append_measurements


def run_measurement_only(
    image_path: str,
    segmentation_path: str,
    output_folder: str,
    ref_position: Optional[int] = None,
    more_info: bool = False,
    slice_index: int = 0,
    color_style: str = "default",
):
    """
    Load an image and a pre-computed segmentation, then run measurements using the same tools
    as in the original annotator.

    Args:
        image_path: Path to the image file (TIF, H5 or VOL).
        segmentation_path: Path to the pre-computed segmentation (TIF or H5).
        output_folder: Folder to save the measurement workbook in.
        ref_position: Horizontal pixel coordinate for reference point (optional).
        more_info: Whether to include additional thickness metrics.
        slice_index: Index of the B-scan in a stack. A file with a single B-scan ignores it.
        color_style: Color style for the visualization in napari.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    image_vol = read_volume(image_path)
    if len(image_vol) == 1:
        slice_index = 0  # A single B-scan has no slice index.
    image = image_vol[slice_index]
    segmentation = read_bscan(segmentation_path, slice_index, SEGMENTATION_KEYS)

    # Ensure segmentation is integer type
    if segmentation.dtype != np.uint32 and segmentation.dtype != np.int32:
        segmentation = segmentation.astype(np.uint32)

    # Create napari viewer
    viewer = napari.Viewer(title="Measurement Only Viewer")

    # Add image and segmentation layers
    viewer.add_image(image, name="Image", colormap="gray", opacity=0.8)
    viewer.add_labels(segmentation, name="Segmentation", opacity=0.8)
    colormap = get_layer_colormap(color_style)
    if colormap is not None:
        viewer.layers["Segmentation"].colormap = colormap

    # Set reference point
    image_shape = image.shape
    central_point = (image_shape[0] // 2, image_shape[1] // 2)
    if ref_position is None:
        ref_point = (image_shape[0] // 2, image_shape[1] // 3)
    else:
        ref_position = max(0, min(ref_position, image_shape[1] - 1))
        ref_point = (image_shape[0] // 2, ref_position)

    # Add reference points
    viewer.add_points(central_point, name="fovea reference point", face_color="white", size=10, visible=True)
    viewer.add_points(ref_point, name="thickness reference point", face_color="blue", size=10, visible=True)

    # Add line shapes for length measurement
    viewer.add_shapes(
        name="Lines",
        shape_type="line",
        edge_color="red",
        edge_width=2,
    )

    # Add measurement widgets. The "Measure" button is the only thing that writes a measurement.
    save_fn = partial(
        append_measurements,
        output_folder=output_folder,
        source_name=basename,
        slice_index=slice_index,
    )
    measurement_widget = MeasurementTableWidget(viewer, _measure, more_info, layer_name="Segmentation",
                                                save_fn=save_fn)
    viewer.window.add_dock_widget(measurement_widget, name="Measurement Table", area="right")

    line_length_widget = LineLengthTableWidget(viewer)
    viewer.window.add_dock_widget(line_length_widget, name="Line Length Measurer", area="right")

    colormap_widget = ColormapWidget(viewer)
    viewer.window.add_dock_widget(colormap_widget, name="Label Color Map", area="right")

    # Run napari
    napari.run()
