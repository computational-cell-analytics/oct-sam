"""Tests for oct_tools.napari_widgets.utils.save_measurements.

No napari viewer is created. save_measurements only needs a `layers` mapping that supports `in`
and `[]`, so a small stand-in is enough and the tests run headless.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from oct_tools.napari_widgets.utils import save_measurements

HEIGHT, WIDTH = 60, 200
FOVEA_LAYER = "fovea reference point"
REFERENCE_LAYER = "thickness reference point"


class FakeLayer:
    def __init__(self, data):
        self.data = data


class FakeViewer:
    """Stand-in for napari.Viewer, exposing only the `layers` mapping save_measurements uses."""

    def __init__(self, layers):
        self.layers = layers


def _make_segmentation() -> np.ndarray:
    seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    seg[10:20, :] = 1
    seg[20:35, :] = 2
    seg[35:41, :] = 3
    return seg


class TestSaveMeasurements(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_folder = self.tmp_dir.name
        self.segmentation = FakeLayer(_make_segmentation())

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _viewer(self, fovea=((0, 100),), reference=((0, 100),), with_segmentation=True):
        layers = {}
        if with_segmentation:
            layers["Segmentation"] = self.segmentation
        if fovea is not None:
            layers[FOVEA_LAYER] = FakeLayer(np.array(fovea) if len(fovea) else np.empty((0, 2)))
        if reference is not None:
            layers[REFERENCE_LAYER] = FakeLayer(np.array(reference) if len(reference) else np.empty((0, 2)))
        return FakeViewer(layers)

    def _save(self, viewer, name="scan"):
        save_measurements(viewer, name, self.output_folder, segmentation_layer_name="Segmentation")
        return sorted(os.listdir(self.output_folder))

    def test_writes_table_with_both_points(self):
        files = self._save(self._viewer())
        self.assertEqual(files, ["scan_measurement_00.tsv"])
        table = pd.read_csv(os.path.join(self.output_folder, files[0]), sep="\t")
        self.assertIn("thickness@100px[µm]", table.columns)
        self.assertIn("CFT@100px[µm]", table.columns)
        self.assertEqual(len(table), 3)

    def test_empty_thickness_layer_does_not_crash(self):
        # Regression test: ref_point used to be left unassigned here, raising UnboundLocalError.
        files = self._save(self._viewer(reference=()))
        self.assertEqual(files, ["scan_measurement_00.tsv"])
        table = pd.read_csv(os.path.join(self.output_folder, files[0]), sep="\t")
        self.assertFalse([c for c in table.columns if c.startswith("thickness@")])
        self.assertIn("CFT@100px[µm]", table.columns)

    def test_empty_fovea_layer_omits_etdrs_columns(self):
        files = self._save(self._viewer(fovea=()))
        table = pd.read_csv(os.path.join(self.output_folder, files[0]), sep="\t")
        self.assertFalse([c for c in table.columns if c.startswith("CFT@")])
        self.assertNotIn("central_area[mm²]", table.columns)
        self.assertIn("thickness@100px[µm]", table.columns)

    def test_missing_point_layers_do_not_crash(self):
        # A deleted layer used to raise KeyError before the membership check was added.
        files = self._save(self._viewer(fovea=None, reference=None))
        self.assertEqual(files, ["scan_measurement_00.tsv"])
        table = pd.read_csv(os.path.join(self.output_folder, files[0]), sep="\t")
        self.assertEqual(len(table), 3)

    def test_only_the_first_point_is_used(self):
        viewer = self._viewer(reference=((0, 100), (0, 20)))
        files = self._save(viewer)
        table = pd.read_csv(os.path.join(self.output_folder, files[0]), sep="\t")
        self.assertIn("thickness@100px[µm]", table.columns)
        self.assertNotIn("thickness@20px[µm]", table.columns)

    def test_output_file_index_increments(self):
        viewer = self._viewer()
        self._save(viewer)
        files = self._save(viewer)
        self.assertEqual(files, ["scan_measurement_00.tsv", "scan_measurement_01.tsv"])

    def test_missing_segmentation_layer_returns_early(self):
        files = self._save(self._viewer(with_segmentation=False))
        self.assertEqual(files, [])


if __name__ == "__main__":
    unittest.main()
