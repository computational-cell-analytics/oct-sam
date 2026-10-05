"""Tests for oct_tools.metric_utils.

The tests pin the spacing convention: the first spacing value is the vertical pitch, across the
retinal layers, and the second is the horizontal pitch, along them. A swapped axis order changes
every thickness and every ETDRS area, so these tests fail loudly if the order is reversed again.
"""
import contextlib
import io
import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from oct_tools.metric_utils import (
    VOXEL_SIZE, binary_thickness, calculate_metrics, measure_binary_thickness, run_measurement,
)

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


class TestTotalCentralFovealThickness(unittest.TestCase):
    """The total CFT spans the whole retina, from the top of the RNFL to the bottom of the RPE."""

    def setUp(self):
        self.fovea_column = WIDTH // 2
        self.column = f"CFT_total@{self.fovea_column}px[µm]"

    def _table(self, seg):
        return run_measurement(seg, fovea_point=[0, self.fovea_column])

    def test_matches_the_sum_without_gaps(self):
        table = self._table(_make_segmentation())
        per_layer = table[f"CFT@{self.fovea_column}px[µm]"].sum()
        self.assertAlmostEqual(table[self.column].iloc[0], per_layer, places=6)
        self.assertAlmostEqual(table[self.column].iloc[0], sum(LAYER_ROWS.values()) * SPACING_Y, places=6)

    def test_includes_an_empty_layer(self):
        # Remove the middle layer. Its rows are now unlabeled, but they still lie within the retina.
        seg = _make_segmentation()
        seg[seg == 2] = 0
        table = self._table(seg)
        per_layer = table[f"CFT@{self.fovea_column}px[µm]"].sum()
        expected = sum(LAYER_ROWS.values()) * SPACING_Y

        self.assertAlmostEqual(table[self.column].iloc[0], expected, places=6)
        self.assertAlmostEqual(per_layer, (LAYER_ROWS[1] + LAYER_ROWS[3]) * SPACING_Y, places=6)
        self.assertGreater(table[self.column].iloc[0], per_layer)

    def test_is_the_same_for_every_row(self):
        table = self._table(_make_segmentation())
        self.assertEqual(table[self.column].nunique(), 1)

    def test_notification_reports_the_span(self):
        from oct_tools.metric_utils import get_etdrs_mask
        seg = _make_segmentation()
        seg[seg == 2] = 0
        _, notification = get_etdrs_mask(seg, fovea_point=[0, self.fovea_column])
        expected = round(sum(LAYER_ROWS.values()) * SPACING_Y, 2)
        self.assertIn(str(expected), notification)


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


def _make_binary_segmentation() -> np.ndarray:
    """Build a B-scan with an empty column and columns of 2, 4 and 6 retina pixels."""
    seg = np.zeros((8, 4), dtype=np.uint32)
    seg[2:4, 1] = 1
    seg[2:6, 2] = 2
    seg[1:4, 3] = 1
    seg[4:7, 3] = 2
    return seg


class TestBinaryThickness(unittest.TestCase):
    """Every label ID above 0 is retina, and columns without retina are not used."""

    def test_statistics(self):
        thickness = binary_thickness(_make_binary_segmentation(), SPACING_Y)
        expected = {
            "mean_thickness[µm]": 4 * SPACING_Y,
            "stdev_thickness[µm]": np.sqrt(8 / 3) * SPACING_Y,
            "median_thickness[µm]": 4 * SPACING_Y,
            "min_thickness[µm]": 2 * SPACING_Y,
            "max_thickness[µm]": 6 * SPACING_Y,
        }
        self.assertEqual(thickness.keys(), expected.keys())
        for column, value in expected.items():
            self.assertAlmostEqual(thickness[column], value, places=6, msg=column)

    def test_no_retina(self):
        self.assertEqual(binary_thickness(np.zeros((8, 4), dtype=np.uint32)), {})


class TestMeasureBinaryThickness(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.image_dir, self.label_dir = (os.path.join(self.tmp_dir.name, d) for d in ("images", "labels"))
        os.makedirs(self.image_dir)
        os.makedirs(self.label_dir)
        self.output_path = os.path.join(self.tmp_dir.name, "thickness.xlsx")

        bscan = np.zeros((8, 4), dtype=np.uint8)
        imageio.imwrite(os.path.join(self.image_dir, "a.tif"), np.stack([bscan] * 3, axis=-1))
        imageio.imwrite(os.path.join(self.image_dir, "b.tif"), bscan)
        imageio.imwrite(os.path.join(self.image_dir, "c.tif"), bscan)
        imageio.imwrite(os.path.join(self.label_dir, "a.tif"), _make_binary_segmentation())
        label_b = np.zeros((8, 4), dtype=np.uint32)
        label_b[2:5] = 1
        imageio.imwrite(os.path.join(self.label_dir, "b.tif"), label_b)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_directories_update_the_workbook(self):
        old = pd.DataFrame({"file": ["z.tif", "a.tif"], "mean_thickness[µm]": [1.0, 999.0]})
        old.to_excel(self.output_path, index=False)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            measure_binary_thickness(self.image_dir, self.label_dir, self.output_path, SPACING_Y)
        self.assertIn("['a.tif']", output.getvalue())

        table = pd.read_excel(self.output_path).set_index("file")
        self.assertEqual(list(table.index), ["a.tif", "b.tif", "z.tif"])
        self.assertAlmostEqual(table.loc["a.tif", "mean_thickness[µm]"], 4 * SPACING_Y, places=6)
        self.assertAlmostEqual(table.loc["b.tif", "max_thickness[µm]"], 3 * SPACING_Y, places=6)
        self.assertEqual(table.loc["z.tif", "mean_thickness[µm]"], 1.0)

    def test_single_files_create_the_workbook(self):
        measure_binary_thickness(
            os.path.join(self.image_dir, "b.tif"), os.path.join(self.label_dir, "a.tif"), self.output_path,
        )
        table = pd.read_excel(self.output_path)
        self.assertEqual(list(table["file"]), ["b.tif"])
        self.assertAlmostEqual(table["min_thickness[µm]"][0], 2 * VOXEL_SIZE[1], places=6)

    def test_invalid_input_raises(self):
        image = os.path.join(self.image_dir, "b.tif")
        wide_image = os.path.join(self.tmp_dir.name, "wide.tif")
        imageio.imwrite(wide_image, np.zeros((8, 5), dtype=np.uint8))
        label = os.path.join(self.label_dir, "a.tif")
        cases = {
            "file and directory": (image, self.label_dir, self.output_path),
            "shape mismatch": (wide_image, label, self.output_path),
            "no xlsx": (image, label, os.path.join(self.tmp_dir.name, "thickness.xls")),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                measure_binary_thickness(*args)

        imageio.imwrite(os.path.join(self.label_dir, "d.tif"), _make_binary_segmentation())
        with self.subTest("label without image"), self.assertRaisesRegex(ValueError, "without an image"):
            measure_binary_thickness(self.image_dir, self.label_dir, self.output_path)

        os.remove(os.path.join(self.label_dir, "d.tif"))
        imageio.imwrite(os.path.join(self.image_dir, "s.tif"), np.zeros((3, 8, 4), dtype=np.uint8))
        imageio.imwrite(os.path.join(self.label_dir, "s_z000.tif"), _make_binary_segmentation())
        with self.subTest("stack"), self.assertRaisesRegex(ValueError, "stacks"):
            measure_binary_thickness(self.image_dir, self.label_dir, self.output_path)


if __name__ == "__main__":
    unittest.main()
