import os
import napari
import numpy as np
import pandas as pd

from qtpy.QtWidgets import QDockWidget, QPushButton

from oct_tools.metric_utils import run_measurement, get_etdrs_mask
from oct_tools.layer_information import identify_layers_naively


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
    layer_mapping = identify_layers_naively(segmentation, generic_names=True)
    if layer_mapping is None:
        unique_ids = np.unique(segmentation)[1:]
        layer_mapping = pd.DataFrame(dict(label_id=unique_ids, layer=unique_ids))
    else:
        layer_mapping = pd.DataFrame(dict(label_id=layer_mapping.keys(), layer=layer_mapping.values()))
    measurements = run_measurement(
        segmentation, extra_columns=layer_mapping, fovea_point=fovea_point, reference_point=reference_point,
        extra_information=extra_information,
    )
    etdrs_mask, notification_str = get_etdrs_mask(segmentation, measurements, fovea_point=fovea_point)
    # Reorder the columns so that the layer name is the second column.
    cols = measurements.columns.values.tolist()
    new_col_order = cols[-1:] + cols[:1] + cols[1:-1]
    measurements = measurements[new_col_order]
    measurements = measurements.sort_values("layer").reset_index(drop=True).copy()
    print(measurements)
    return measurements, etdrs_mask, notification_str


def save_measurements(
    viewer: napari.Viewer,
    reference_name: str,
    output_folder,
    segmentation_layer_name: str = "Segmentation",
    more_info: bool = False,
):
    """Save measurement table in an output folder.
    Checks for 'fovea reference point' and 'thickness reference point' layers.
    Only takes the first point in each of these layers.

    Args:
        viewer: Napari viewer.
        reference_name: Name prefix for output file.
        output_folder: Output folder
        segmentation_layer_name: Name of layer in which the segmentation is located.
        more_info: Add additional global information about retinal layers like length, max, min, and mean thickness.
    """
    # Get the segmentation layer
    if segmentation_layer_name not in viewer.layers:
        napari.utils.notifications.show_error(f"No {segmentation_layer_name} layer found.")
        return
    segmentation = viewer.layers[segmentation_layer_name].data

    # Get the fovea reference point layer
    fovea_layer = viewer.layers["fovea reference point"]
    if fovea_layer is None or len(fovea_layer.data) == 0:
        napari.utils.notifications.show_warning("No fovea reference point found.")
        fovea_point = None
    else:
        fovea_point = tuple(fovea_layer.data[0])  # First point only

    # Get the thickness reference point layer
    ref_layer = viewer.layers["thickness reference point"]
    if ref_layer is None or len(ref_layer.data) == 0:
        napari.utils.notifications.show_warning("No thickness reference point found.")
    else:
        ref_point = tuple(ref_layer.data[0])  # First point only

    # Run measurement with current point positions
    measurements, _, _ = _measure(segmentation, fovea_point=fovea_point, reference_point=ref_point,
                                  extra_information=more_info)

    # Save to file
    i = len([f for f in os.listdir(output_folder) if
             f.startswith(f"{reference_name}_measurement_") and
             f.endswith(".tsv")])
    output_path = os.path.join(output_folder, f"{reference_name}_measurement_{i:02}.tsv")
    measurements.to_csv(output_path, sep="\t", index=False)
    napari.utils.notifications.show_info(f"Measurements saved to {output_path}")
