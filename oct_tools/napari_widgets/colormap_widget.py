import napari
from qtpy.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from oct_tools.layer_information import get_layer_colormap


class ColormapWidget(QWidget):
    """Dock widget for interactively applying a label color style to napari label layers."""

    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self._viewer = viewer

        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)

        layout.addWidget(QLabel("Color style:"))

        self._combo = QComboBox()
        self._combo.addItems(["default", "custom", "random", "check"])
        layout.addWidget(self._combo)

        btn = QPushButton("Apply to selected layer(s)")
        btn.clicked.connect(self._apply)
        layout.addWidget(btn)

        layout.addStretch()
        self.setLayout(layout)

    def _apply(self):
        style = self._combo.currentText()
        colormap = get_layer_colormap(style)

        targets = [
            layer for layer in self._viewer.layers.selection
            if isinstance(layer, napari.layers.Labels)
        ]
        if not targets:
            napari.utils.notifications.show_warning(
                "Select one or more Labels layers before applying."
            )
            return

        for layer in targets:
            if colormap is not None:
                layer.colormap = colormap
            else:
                layer.new_colormap()

        napari.utils.notifications.show_info(
            f"Applied '{style}' colormap to {len(targets)} layer(s)."
        )

    def apply_style(self, style: str):
        """Programmatically apply a style to all Labels layers in the viewer."""
        idx = self._combo.findText(style)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._apply()
