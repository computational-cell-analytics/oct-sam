"""Tests for oct_tools.metric_utils.

The tests pin the spacing convention: the first spacing value is the vertical pitch, across the
retinal layers, and the second is the horizontal pitch, along them. A swapped axis order changes
every thickness and every ETDRS area, so these tests fail loudly if the order is reversed again.
"""
import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np

from oct_tools.metric_utils import VOXEL_SIZE, calculate_metrics, run_measurement

# Default pixel spacing of the UMG-RP data, in micrometer.
SPACING_Y = 3.87166976  # Vertical, across the retinal layers.
SPACING_X = 5.8814      # Horizontal, along the retinal layers.

HEIGHT, WIDTH = 60, 200

# Rows occupied by each layer of the synthetic B-scan built in _make_segmentation.
LAYER_ROWS = {1: 10, 2: 15, 3: 6}


def _make_segmentation() -> np.ndarray:
    """Build a synthetic B-scan of stacked, full-width retinal layers."""
    seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    seg[10:20, :] = 1
    seg[20:35, :] = 2
    seg[35:41, :] = 3
    return seg


class TestSpacingConvention(unittest.TestCase):
    """The vertical spacing scales thickness, the horizontal spacing scales along-layer distance."""

    def setUp(self):
        self.seg = _make_segmentation()
        self.table = run_measurement(self.seg, extra_information=True).set_index("label_id")

    def test_default_spacing_is_vertical_then_horizontal(self):
        self.assertEqual(VOXEL_SIZE[1:], (SPACING_Y, SPACING_X))

    def test_thickness_uses_vertical_spacing(self):
        for label_id, rows in LAYER_ROWS.items():
            expected = rows * SPACING_Y
            for column in ("max_thickness[µm]", "min_thickness[µm]", "mean_thickness[µm]"):
                self.assertAlmostEqual(self.table.loc[label_id, column], expected, places=6)

    def test_area_uses_both_spacings(self):
        for label_id, rows in LAYER_ROWS.items():
            expected = rows * WIDTH * SPACING_Y * SPACING_X / 1e6
            self.assertAlmostEqual(self.table.loc[label_id, "area[mm²]"], expected, places=9)

    def test_length_uses_horizontal_spacing(self):
        # A layer one pixel high spans the full width, so its centerline length is set by the
        # horizontal spacing alone.
        seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
        seg[25:26, :] = 1
        table = run_measurement(seg, extra_information=True)
        self.assertAlmostEqual(table["length[µm]"][0], (WIDTH - 1) * SPACING_X, places=6)


class TestReferencePoint(unittest.TestCase):
    """Thickness at a single column, including a layer that is thinner there."""

    def setUp(self):
        # Layer 1 is 10 rows high, except over columns 50-59 where it is notched down to 6 rows.
        self.seg = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
        self.seg[10:20, :] = 1
        self.seg[10:14, 50:60] = 0

    def test_thickness_at_reference_column(self):
        table = run_measurement(self.seg, extra_information=True, reference_point=[0, 55])
        self.assertAlmostEqual(table["max_thickness[µm]"][0], 10 * SPACING_Y, places=6)
        self.assertAlmostEqual(table["min_thickness[µm]"][0], 6 * SPACING_Y, places=6)
        self.assertAlmostEqual(table["thickness@55px[µm]"][0], 6 * SPACING_Y, places=6)

    def test_thickness_outside_the_notch(self):
        table = run_measurement(self.seg, extra_information=True, reference_point=[0, 10])
        self.assertAlmostEqual(table["thickness@10px[µm]"][0], 10 * SPACING_Y, places=6)

    def test_reference_point_outside_raises(self):
        with self.assertRaises(ValueError):
            run_measurement(self.seg, reference_point=[0, WIDTH + 1])


class TestEtdrsGrid(unittest.TestCase):
    """The ETDRS radii are converted into column offsets with the horizontal spacing."""

    def setUp(self):
        self.seg = _make_segmentation()
        self.fovea_column = WIDTH // 2

    def test_central_band_width_uses_horizontal_spacing(self):
        table = run_measurement(self.seg, fovea_point=[0, self.fovea_column]).set_index("label_id")
        # The central band reaches 500 µm to either side of the fovea column.
        half_width = round(500 / SPACING_X)
        n_columns = 2 * half_width + 1
        for label_id, rows in LAYER_ROWS.items():
            expected = rows * n_columns * SPACING_Y * SPACING_X / 1e6
            self.assertAlmostEqual(table.loc[label_id, "central_area[mm²]"], expected, places=9)

    def test_central_foveal_thickness(self):
        table = run_measurement(self.seg, fovea_point=[0, self.fovea_column]).set_index("label_id")
        column = f"CFT@{self.fovea_column}px[µm]"
        for label_id, rows in LAYER_ROWS.items():
            self.assertAlmostEqual(table.loc[label_id, column], rows * SPACING_Y, places=6)


class TestCalculateMetrics(unittest.TestCase):
    """The oct_tools.metrics CLI path must agree with the napari path.

    Regression test for the reversed voxel size: calculate_metrics used to pass
    np.array(voxel_size)[::-1] to run_measurement, which made every thickness larger by
    SPACING_X / SPACING_Y and shifted the ETDRS band edges.
    """

    def setUp(self):
        self.seg = _make_segmentation()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmp_dir.name, "segmentation.tif")
        self.output_path = os.path.join(self.tmp_dir.name, "measurement.tsv")
        imageio.imwrite(self.input_path, self.seg)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _run_cli(self, voxel_size, **kwargs):
        import pandas as pd
        calculate_metrics(self.input_path, self.output_path, voxel_size, **kwargs)
        return pd.read_csv(self.output_path, sep="\t")

    def test_matches_run_measurement(self):
        cli_table = self._run_cli([SPACING_Y, SPACING_X], reference_position=100, fovea_position=100)
        expected = run_measurement(
            self.seg, extra_information=True, reference_point=[0, 100], fovea_point=[0, 100],
        )
        self.assertEqual(list(cli_table.columns), list(expected.columns))
        for column in expected.columns:
            np.testing.assert_allclose(
                cli_table[column].to_numpy(), expected[column].to_numpy(),
                rtol=1e-9, atol=1e-9, err_msg=f"column {column} differs",
            )

    def test_thickness_matches_vertical_spacing(self):
        cli_table = self._run_cli([SPACING_Y, SPACING_X]).set_index("label_id")
        for label_id, rows in LAYER_ROWS.items():
            self.assertAlmostEqual(cli_table.loc[label_id, "max_thickness[µm]"], rows * SPACING_Y, places=6)

    def test_single_voxel_size_is_used_for_both_axes(self):
        cli_table = self._run_cli([SPACING_Y]).set_index("label_id")
        for label_id, rows in LAYER_ROWS.items():
            self.assertAlmostEqual(cli_table.loc[label_id, "max_thickness[µm]"], rows * SPACING_Y, places=6)
            expected_area = rows * WIDTH * SPACING_Y * SPACING_Y / 1e6
            self.assertAlmostEqual(cli_table.loc[label_id, "area[mm²]"], expected_area, places=9)

    def test_etdrs_grid_is_exported(self):
        etdrs_path = os.path.join(self.tmp_dir.name, "etdrs.tif")
        self._run_cli([SPACING_Y, SPACING_X], fovea_position=100, etdrs_grid=etdrs_path)
        self.assertTrue(os.path.exists(etdrs_path))
        mask = imageio.imread(etdrs_path)
        self.assertEqual(mask.shape, self.seg.shape)
        # The mask holds 1 for the central band, 2 for the inner ring and 3 for the outer ring.
        self.assertTrue(set(np.unique(mask)).issubset({0, 1, 2, 3}))

    def test_etdrs_grid_without_fovea_raises(self):
        etdrs_path = os.path.join(self.tmp_dir.name, "etdrs.tif")
        with self.assertRaises(ValueError):
            self._run_cli([SPACING_Y, SPACING_X], etdrs_grid=etdrs_path)


if __name__ == "__main__":
    unittest.main()
