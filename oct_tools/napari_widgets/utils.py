import os
import re
from typing import Callable, Optional, Tuple, Union

import napari
import numpy as np
import pandas as pd

from oct_tools.metric_utils import run_measurement, get_etdrs_mask
from oct_tools.layer_information import identify_layers_naively

MEASUREMENT_FILE = "measurements.xlsx"

# Column names carry the position they were measured at, e.g. "CFT@103px[µm]". The position is
# split off into its own column so that a whole session shares one set of column names.
_POSITION_COLUMN = re.compile(r"^(?P<name>.+)@(?P<position>[0-9.]+)px(?P<unit>\[.*\])$")
_POSITION_COLUMNS = {"CFT": "fovea_x", "CFT_total": "fovea_x", "thickness": "ref_x"}


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
    etdrs_mask, notification_str = get_etdrs_mask(segmentation, fovea_point=fovea_point)
    # Reorder the columns so that the layer name is the second column.
    cols = measurements.columns.values.tolist()
    new_col_order = cols[-1:] + cols[:1] + cols[1:-1]
    measurements = measurements[new_col_order]
    measurements = measurements.sort_values("layer").reset_index(drop=True).copy()
    print(measurements)
    return measurements, etdrs_mask, notification_str


def _first_point(viewer: napari.Viewer, layer_name: str) -> Optional[Tuple[float]]:
    """Return the first point of a napari point layer.

    A missing or empty layer is not an error. It means that the caller skips the columns which
    depend on that point.

    Args:
        viewer: Napari viewer.
        layer_name: Name of the point layer.

    Returns:
        The first point as (row, column), or None.
    """
    if layer_name not in viewer.layers or len(viewer.layers[layer_name].data) == 0:
        napari.utils.notifications.show_warning(f"No {layer_name} found.")
        return None
    points = viewer.layers[layer_name].data
    if len(points) > 1:
        napari.utils.notifications.show_warning(f"More than one point in layer {layer_name}. Taking the first one.")
    return tuple(points[0])


def read_measurement_inputs(
    viewer: napari.Viewer,
    segmentation_layer_name: str = "Segmentation",
    fovea_layer: str = "fovea reference point",
    ref_layer: str = "thickness reference point",
):
    """Read the segmentation and the two reference points from the viewer.

    Args:
        viewer: Napari viewer.
        segmentation_layer_name: Name of layer in which the segmentation is located.
        fovea_layer: Name of the point layer holding the foveal reference point.
        ref_layer: Name of the point layer holding the thickness reference point.

    Returns:
        (segmentation, fovea_point, reference_point). The segmentation is None if its layer is
        missing. Either point is None if its layer is missing or empty.
    """
    if segmentation_layer_name not in viewer.layers:
        napari.utils.notifications.show_error(f"No {segmentation_layer_name} layer found.")
        return None, None, None
    segmentation = viewer.layers[segmentation_layer_name].data
    return segmentation, _first_point(viewer, fovea_layer), _first_point(viewer, ref_layer)


def _split_positions(measurements: pd.DataFrame) -> pd.DataFrame:
    """Move the measurement position out of the column names into its own column."""
    positions = {}
    renamed = {}
    for column in measurements.columns:
        match = _POSITION_COLUMN.match(column)
        if match is None or match["name"] not in _POSITION_COLUMNS:
            continue
        renamed[column] = f"{match['name']}{match['unit']}"
        positions[_POSITION_COLUMNS[match["name"]]] = float(match["position"])
    measurements = measurements.rename(columns=renamed)
    for name in ("ref_x", "fovea_x"):
        if name in positions:
            measurements.insert(0, name, positions[name])
    return measurements


def append_measurements(
    measurements: pd.DataFrame,
    output_folder: str,
    source_name: str,
    slice_index: Union[int, Callable[[], int]] = 0,
):
    """Append one measurement table to the workbook in an output folder.

    All measurements of a session accumulate in a single file. Every row records the source file,
    the B-scan index and the positions the measurement was taken at.

    Args:
        measurements: Measurement table for one B-scan.
        output_folder: Output folder.
        source_name: Name of the file the measurement was taken from.
        slice_index: Index of the measured B-scan, or a callable returning it.
    """
    if callable(slice_index):
        slice_index = slice_index()

    measurements = _split_positions(measurements.copy())
    measurements.insert(0, "z", slice_index)
    measurements.insert(0, "file", source_name)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, MEASUREMENT_FILE)
    if os.path.exists(output_path):
        measurements = pd.concat([pd.read_excel(output_path), measurements], ignore_index=True)
    measurements.to_excel(output_path, index=False)
    napari.utils.notifications.show_info(f"Measurements saved to {output_path}")
