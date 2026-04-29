import os
from typing import Optional

import numpy as np
import napari
import pandas as pd
from qtpy.QtWidgets import QPushButton
from imageio.v3 import imread
from h5py import File

from oct_tools.metric_utils import run_measurement, get_etdrs_mask
from oct_tools.layer_information import identify_layers_naively
from oct_tools.napari_widgets.table_widget import MeasurementTableWidget
from oct_tools.napari_widgets.linelength_widget import LineLengthTableWidget


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

    # Define measurement function (same as in original)
    def _measure(segmentation, fovea_point=None, reference_point=None, extra_information=False):
        layer_mapping = identify_layers_naively(segmentation, generic_names=True)
        if layer_mapping is None:
            unique_ids = np.unique(segmentation)[1:]
            layer_mapping = pd.DataFrame(dict(label_id=unique_ids, layer=unique_ids))
        else:
            layer_mapping = pd.DataFrame(dict(label_id=layer_mapping.keys(), layer=layer_mapping.values()))
        measurements = run_measurement(
            segmentation, extra_columns=layer_mapping, fovea_point=fovea_point,
            reference_point=reference_point, extra_information=extra_information
        )
        etdrs_mask, notification_str = get_etdrs_mask(segmentation, measurements, fovea_point=fovea_point)
        # Reorder columns
        cols = measurements.columns.values.tolist()
        new_col_order = cols[-1:] + cols[:1] + cols[1:-1]
        measurements = measurements[new_col_order]
        measurements = measurements.sort_values("layer").reset_index(drop=True).copy()
        print(measurements)
        return measurements, etdrs_mask, notification_str

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

    # Add a button to save measurements
    def save_measurements():
        segmentation_layer = viewer.layers["Segmentation"].data
        measurements, _, _ = _measure(segmentation_layer, fovea_point=central_point,
                                      reference_point=ref_point, extra_information=more_info)
        i = len([f for f in os.listdir(output_folder) if f.startswith("measurement_") and f.endswith(".tsv")])
        output_path = os.path.join(output_folder, f"measurement_{i:05}.tsv")
        measurements.to_csv(output_path, sep="\t", index=False)
        napari.utils.notifications.show_info(f"Measurements saved to {output_path}")

    # Add a button to trigger measurement saving
    save_button = QPushButton("Save Measurements")
    save_button.clicked.connect(save_measurements)
    viewer.window.add_dock_widget(save_button, name="Save Measurements", area="bottom")

    # Run napari
    napari.run()
