from __future__ import annotations

import io
import pandas as pd
import matplotlib.pyplot as plt

from magicgui import magicgui
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget, QScrollArea

import napari


def df_to_png_bytes(df: pd.DataFrame, *, dpi: int = 200, fontsize: int = 10) -> bytes:
    """Render a pandas dataframe as a PNG (in-memory)."""
    # Heuristic sizing: scale with rows/cols so it stays readable.
    nrows, ncols = df.shape
    fig_w = max(4.0, 1.4 * ncols)
    fig_h = max(1.5, 0.35 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Convert to strings for stable display (esp. floats)
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


class MeasurementTableWidget(QWidget):
    """
    A QWidget that hosts a magicgui function + an image preview below it.
    Provide your measurement function via `measure_fn`.
    """

    def __init__(self, viewer: napari.Viewer, measure_fn,
                 layer_name="committed_objects", point_name="fovea reference point"):
        super().__init__()
        self._viewer = viewer
        self._measure_fn = measure_fn
        self._layer_name = layer_name
        self._point_name = point_name

        self._img_label = QLabel("OCT Measurements")
        self._img_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._img_label.setMinimumHeight(200)
        self._img_label.setStyleSheet("QLabel { background: #222; color: #ddd; }")
        self._img_label.setBackgroundRole(self._img_label.backgroundRole())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignCenter)
        self._scroll.setWidget(self._img_label)

        @magicgui(call_button="Measure")
        def gui():
            labels = self._viewer.layers[self._layer_name].data
            points = self._viewer.layers[self._point_name].data

            if len(points) == 0:
                reference_point = None
            else:
                reference_point = list(points[0])
                if len(points) > 1:
                    print(f"More than one point in layer {point_name}. Taking the first one.")

            # ---- your existing measurement logic call ----
            # Must return a pandas.DataFrame.
            df, etdrs_mask, notification_str = self._measure_fn(labels, reference_point=reference_point)
            df = df.round(2)

            # present optional ETDRS sections (central, inner ring, outer ring)
            if etdrs_mask is not None:
                self._viewer.add_labels(etdrs_mask)

            # output central foveal thickness or additional notification strings
            if notification_str is not None:
                napari.utils.notifications.show_info(notification_str)

            dpi = 100
            fontsize = 8
            png = df_to_png_bytes(df, dpi=dpi, fontsize=fontsize)

            pix = QPixmap()
            pix.loadFromData(png, "PNG")
            self._img_label.setPixmap(pix)
            self._img_label.adjustSize()  # IMPORTANT

        self._gui = gui

        layout = QVBoxLayout()
        layout.addWidget(self._gui.native)
        layout.addWidget(self._scroll)
        self.setLayout(layout)
