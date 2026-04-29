import os
from functools import partial
from typing import Optional

import numpy as np
import napari
from qtpy.QtWidgets import QPushButton
from imageio.v3 import imread
from h5py import File

from oct_tools.napari_widgets.table_widget import MeasurementTableWidget
from oct_tools.napari_widgets.linelength_widget import LineLengthTableWidget
from oct_tools.napari_widgets.utils import _measure, save_measurements


def run_measurement_only(
    image_path: str,
    segmentation_path: str,
    output_folder: str,
    ref_position: Optional[int] = None,
    more_info: bool = False,
    slice_index: int = 0,
):
    """
    Load an image and a pre-computed segmentation, then run measurements using the same tools
    as in the original annotator.

    Args:
        image_path: Path to the image file (TIF or H5).
        segmentation_path: Path to the pre-computed segmentation (TIF or H5).
        output_folder: Folder to save measurement results (TSV files).
        ref_position: Horizontal pixel coordinate for reference point (optional).
        more_info: Whether to include additional thickness metrics.
        slice_index: Index of slice to load if input is 3D (only used for TIF/H5 3D).
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Load image
    if image_path.endswith(".h5"):
        with File(image_path, "r") as f:
            if "image" in f:
                image = f["image"][:]
            else:
                raise KeyError("H5 file must contain 'image' dataset.")
    elif image_path.endswith(".tif"):
        image_vol = imread(image_path)
        if image_vol.ndim == 3:
            image = image_vol[slice_index]
        elif image_vol.ndim == 2:
            image = image_vol
        else:
            raise ValueError("Image must be 2D or 3D.")
    else:
        raise ValueError("Unsupported image format. Use .tif or .h5.")

    # Load segmentation
    if segmentation_path.endswith(".h5"):
        with File(segmentation_path, "r") as f:
            if "segmentation" in f:
                segmentation = f["segmentation"][:]
            elif "seg" in f:
                segmentation = f["seg"][:]
            else:
                raise KeyError("H5 file must contain 'segmentation' or 'seg' dataset.")
    elif segmentation_path.endswith(".tif"):
        segmentation = imread(segmentation_path)
    else:
        raise ValueError("Unsupported segmentation format. Use .tif or .h5.")

    # Ensure segmentation is integer type
    if segmentation.dtype != np.uint32 and segmentation.dtype != np.int32:
        segmentation = segmentation.astype(np.uint32)

    # Create napari viewer
    viewer = napari.Viewer(title="Measurement Only Viewer")

    # Add image and segmentation layers
    viewer.add_image(image, name="Image", colormap="gray", opacity=0.8)
    viewer.add_labels(segmentation, name="Segmentation", opacity=0.8)

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

    # Add measurement widgets
    measurement_widget = MeasurementTableWidget(viewer, _measure, more_info, layer_name="Segmentation")
    viewer.window.add_dock_widget(measurement_widget, name="Measurement Table", area="right")

    line_length_widget = LineLengthTableWidget(viewer)
    viewer.window.add_dock_widget(line_length_widget, name="Line Length Measurer", area="right")

    # Add a button to trigger measurement saving
    save_func = partial(
        save_measurements,
        viewer=viewer,
        reference_name=basename,
        output_folder=output_folder,
        segmentation_layer_name="Segmentation",
        more_info=more_info
    )

    save_button = QPushButton("Save Measurements")
    save_button.clicked.connect(save_func)
    viewer.window.add_dock_widget(save_button, name="Save Measurements", area="bottom")

    # Run napari
    napari.run()
