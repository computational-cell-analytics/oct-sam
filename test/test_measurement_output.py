"""Tests for oct_tools.napari_widgets.utils, the measurement output path.

No napari viewer is created. read_measurement_inputs only needs a `layers` mapping that supports
`in` and `[]`, so a small stand-in is enough and the tests run headless.
"""
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from oct_tools.napari_widgets.utils import (
    MEASUREMENT_FILE, _measure, append_measurements, read_measurement_inputs,
)

HEIGHT, WIDTH = 60, 200
FOVEA_LAYER = "fovea reference point"
REFERENCE_LAYER = "thickness reference point"


class FakeLayer:
    def __init__(self, data):
        self.data = data


class FakeViewer:
    """Stand-in for napari.Viewer, exposing only the `layers` mapping the readers use."""

    def __init__(self, layers):
        self.layers = layers


def _make_segmentation() -> np.ndarray:
    seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    seg[10:20, :] = 1
    seg[20:35, :] = 2
    seg[35:41, :] = 3
    return seg


def _viewer(fovea=((0, 100),), reference=((0, 100),), with_segmentation=True):
    layers = {}
    if with_segmentation:
        layers["Segmentation"] = FakeLayer(_make_segmentation())
    if fovea is not None:
        layers[FOVEA_LAYER] = FakeLayer(np.array(fovea) if len(fovea) else np.empty((0, 2)))
    if reference is not None:
        layers[REFERENCE_LAYER] = FakeLayer(np.array(reference) if len(reference) else np.empty((0, 2)))
    return FakeViewer(layers)


class TestReadMeasurementInputs(unittest.TestCase):
    """A missing or empty layer means "skip those columns", never a crash."""

    def test_reads_both_points(self):
        seg, fovea, ref = read_measurement_inputs(_viewer())
        np.testing.assert_array_equal(seg, _make_segmentation())
        self.assertEqual(tuple(fovea), (0, 100))
        self.assertEqual(tuple(ref), (0, 100))

    def test_empty_thickness_layer_does_not_crash(self):
        # Regression test: ref_point used to be left unassigned here, raising UnboundLocalError.
        _, fovea, ref = read_measurement_inputs(_viewer(reference=()))
        self.assertIsNone(ref)
        self.assertIsNotNone(fovea)

    def test_missing_point_layers_do_not_crash(self):
        # A deleted layer used to raise KeyError before the membership check was added.
        seg, fovea, ref = read_measurement_inputs(_viewer(fovea=None, reference=None))
        self.assertIsNotNone(seg)
        self.assertIsNone(fovea)
        self.assertIsNone(ref)

    def test_only_the_first_point_is_used(self):
        _, _, ref = read_measurement_inputs(_viewer(reference=((0, 100), (0, 20))))
        self.assertEqual(tuple(ref), (0, 100))

    def test_missing_segmentation_layer_returns_none(self):
        self.assertEqual(read_measurement_inputs(_viewer(with_segmentation=False)), (None, None, None))


class TestAppendMeasurements(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_folder = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _save(self, name="scan", slice_index=0, fovea=(0, 100), reference=(0, 100)):
        measurements, _, _ = _measure(_make_segmentation(), fovea_point=fovea, reference_point=reference)
        append_measurements(measurements, self.output_folder, name, slice_index)
        return sorted(os.listdir(self.output_folder))

    def _table(self):
        return pd.read_excel(os.path.join(self.output_folder, MEASUREMENT_FILE))

    def test_writes_one_workbook(self):
        self.assertEqual(self._save(), [MEASUREMENT_FILE])
        table = self._table()
        self.assertEqual(list(table.columns[:4]), ["file", "z", "fovea_x", "ref_x"])
        self.assertIn("CFT[µm]", table.columns)
        self.assertIn("CFT_total[µm]", table.columns)
        self.assertIn("thickness[µm]", table.columns)
        self.assertEqual(len(table), 3)

    def test_saves_accumulate_in_one_workbook(self):
        self._save(name="scan_a", slice_index=8)
        self.assertEqual(self._save(name="scan_b", slice_index=12), [MEASUREMENT_FILE])

        table = self._table()
        self.assertEqual(len(table), 6)
        self.assertEqual(sorted(set(table["file"])), ["scan_a", "scan_b"])
        self.assertEqual(sorted(set(table["z"])), [8, 12])

    def test_a_moved_fovea_point_reuses_the_same_columns(self):
        self._save(slice_index=8, fovea=(0, 100))
        self._save(slice_index=12, fovea=(0, 137))

        table = self._table()
        self.assertEqual([c for c in table.columns if c.startswith("CFT")], ["CFT[µm]", "CFT_total[µm]"])
        self.assertEqual(sorted(set(table["fovea_x"])), [100.0, 137.0])
        self.assertFalse(table["CFT[µm]"].isna().any())

    def test_missing_points_omit_their_columns(self):
        self._save(fovea=None, reference=None)
        table = self._table()
        self.assertFalse([c for c in table.columns if c.startswith("CFT")])
        self.assertNotIn("thickness[µm]", table.columns)
        self.assertNotIn("fovea_x", table.columns)

    def test_slice_index_can_be_a_callable(self):
        self._save(slice_index=lambda: 42)
        self.assertEqual(set(self._table()["z"]), {42})

    def test_output_folder_is_created(self):
        self.output_folder = os.path.join(self.tmp_dir.name, "new", "nested")
        self._save()
        self.assertTrue(os.path.exists(os.path.join(self.output_folder, MEASUREMENT_FILE)))


if __name__ == "__main__":
    unittest.main()
