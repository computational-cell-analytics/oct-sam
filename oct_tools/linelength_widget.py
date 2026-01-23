from __future__ import annotations

import io
import math
from typing import Tuple

import napari
import numpy as np
import pandas as pd
from magicgui import magicgui
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame, QScrollArea
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from napari import Viewer
from napari.layers import Shapes
from typing import Optional
import matplotlib.pyplot as plt

from oct_tools.segmentation_utils import VOXEL_SIZE


def df_to_png_bytes(df: pd.DataFrame, *, dpi: int = 200, fontsize: int = 10) -> bytes:
    """Render a pandas dataframe as a PNG (in-memory)."""
    nrows, ncols = df.shape
    fig_w = max(4.0, 1.4 * ncols)
    fig_h = max(1.5, 0.35 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    cell_text = df.astype(str).values.tolist()
    col_labels = list(df.columns.astype(str))

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.2)

    buf = io.BytesIO()
    fig.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    buf.seek(0)
    return buf.getvalue()


class LineLengthTableWidget(QWidget):
    """
    A napari widget that:
    - Monitors a Shapes layer for lines.
    - Automatically detects when lines are drawn.
    - Calculates length in µm.
    - Displays a table of all line lengths.
    - Shows the table as a PNG preview.
    - Outputs summary via napari info.
    """

    def __init__(self, viewer: Viewer, voxel_size=VOXEL_SIZE[1:]):
        super().__init__()
        self._viewer = viewer
        self._shape_layer: Optional[Shapes] = None
        self._layer_name: str = ""
        self._voxel_size: Tuple[float] = voxel_size

        # GUI
        @magicgui(
            layer_name={"label": "Shape Layer Name", "widget_type": "LineEdit"},
            call_button="Start Monitoring",
        )
        def monitor_lines(layer_name: str):
            self._layer_name = layer_name

            # Disconnect previous event
            if self._shape_layer:
                self._shape_layer.events.data.disconnect(self._on_data_change)

            # Check layer exists
            if layer_name not in self._viewer.layers:
                napari.utils.notifications.show_info(f"Layer '{layer_name}' not found.")
                return

            layer = self._viewer.layers[layer_name]
            if not isinstance(layer, Shapes):
                napari.utils.notifications.show_info(f"Layer '{layer_name}' is not a Shapes layer.")
                return

            self._shape_layer = layer

            # Connect to data change
            self._shape_layer.events.data.connect(self._on_data_change)

            napari.utils.notifications.show_info(
                f"Monitoring lines in '{layer_name}' with voxel size"
                f" ({voxel_size[0]:.3f}, {voxel_size[1]:.3f}) µm/pixel."
            )

        self._gui = monitor_lines

        # Table preview
        self._img_label = QLabel("No lines detected yet.")
        self._img_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._img_label.setMinimumHeight(200)
        self._img_label.setStyleSheet("QLabel { background: #222; color: #ddd; padding: 8px; }")
        self._img_label.setFrameStyle(QFrame.StyledPanel)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(QtCore.Qt.AlignCenter)
        self._scroll.setWidget(self._img_label)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self._gui.native)
        layout.addWidget(self._scroll)
        self.setLayout(layout)

    def _on_data_change(self, event):
        """Callback triggered when shape layer data changes."""
        layer = event.source
        shapes = layer.data
        shape_types = layer.shape_type
        voxel_size = self._voxel_size

        # Collect all lines (type 1)
        line_indices = [i for i, t in enumerate(shape_types) if t == "line"]
        if not line_indices:
            self._img_label.setText("No lines found in the shape layer.")
            self._img_label.adjustSize()
            napari.utils.notifications.show_info("No lines detected.")
            return

        # Compute length for each line
        results = []
        for idx in line_indices:
            points = shapes[idx]
            if len(points) < 2:
                continue
            # Euclidean distance between first and last point
            dist_x = np.abs(points[-1][0] - points[0][0])
            dist_y = np.abs(points[-1][1] - points[0][1])
            length_um = math.sqrt((dist_x * voxel_size[0]) ** 2 + (dist_y * voxel_size[1]) ** 2)
            results.append({
                "Line ID": idx,
                "Length (µm)": f"{length_um:.2f}",
            })

        # Create DataFrame
        df = pd.DataFrame(results)
        df = df.round(2)

        # Summary notification
        total_lines = len(results)
        total_length_um = df["Length (µm)"].astype(float).sum()
        napari.utils.notifications.show_info(
            f"Found {total_lines} line(s). Length of last line: {total_length_um:.2f} µm."
        )

        # Render table as PNG
        png_bytes = df_to_png_bytes(df, dpi=100, fontsize=8)

        # Display in label
        pix = self._img_label.pixmap()
        if pix is None or pix.isNull():
            pix = QPixmap()
        pix.loadFromData(png_bytes, "PNG")
        self._img_label.setPixmap(pix)
        self._img_label.adjustSize()
