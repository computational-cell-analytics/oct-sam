import contextlib
import io
import os
import tempfile
import unittest

import h5py
import imageio.v3 as imageio
import numpy as np
import tifffile

from oct_tools.data_reader import SEGMENTATION_KEYS, read_bscan, read_volume
from oct_tools.heyex_vol import read_heyex_vol
from test_heyex_vol import _write_vol


class TestDataReader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.bscan = np.arange(40, dtype=np.uint8).reshape(5, 8)

    def _path(self, name):
        return os.path.join(self.tmp_dir.name, name)

    def test_tif_bscan_and_stack(self):
        imageio.imwrite(self._path("a.tif"), self.bscan)
        imageio.imwrite(self._path("s.tif"), np.stack([self.bscan, self.bscan + 1]))
        np.testing.assert_array_equal(read_volume(self._path("a.tif")), self.bscan[np.newaxis])
        self.assertEqual(read_volume(self._path("s.tif")).shape, (2, 5, 8))

    def test_rgb_becomes_gray(self):
        imageio.imwrite(self._path("rgb.tif"), np.stack([self.bscan] * 3, axis=-1))
        volume = read_volume(self._path("rgb.tif"))
        self.assertEqual(volume.dtype, np.uint8)
        np.testing.assert_array_equal(volume, self.bscan[np.newaxis])

    def test_rgba_becomes_gray(self):
        alpha = np.full_like(self.bscan, 255)
        imageio.imwrite(self._path("rgba.png"), np.stack([self.bscan] * 3 + [alpha], axis=-1))
        np.testing.assert_array_equal(read_volume(self._path("rgba.png")), self.bscan[np.newaxis])

    def test_tif_with_one_series_per_page(self):
        stack = np.stack([self.bscan + z for z in range(3)])
        for bscan in stack:
            tifffile.imwrite(self._path("pages.tif"), bscan, append=True)
        np.testing.assert_array_equal(read_volume(self._path("pages.tif")), stack)

    def test_tif_series_with_different_shapes_raise(self):
        tifffile.imwrite(self._path("mixed.tif"), self.bscan, append=True)
        tifffile.imwrite(self._path("mixed.tif"), self.bscan[:3], append=True)
        with self.assertRaises(ValueError):
            read_volume(self._path("mixed.tif"))

    def test_h5_key_order(self):
        path = self._path("x.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("image", data=self.bscan)
            f.create_dataset("labels/original", data=self.bscan + 1)
        np.testing.assert_array_equal(read_bscan(path, keys=SEGMENTATION_KEYS), self.bscan + 1)
        np.testing.assert_array_equal(read_bscan(path), self.bscan)

    def test_h5_missing_key(self):
        path = self._path("x.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("image", data=self.bscan)
        with self.assertRaises(KeyError):
            read_volume(path, SEGMENTATION_KEYS)

    def test_h5_group_is_not_a_dataset(self):
        path = self._path("x.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("labels/original", data=self.bscan)
        with self.assertRaises(KeyError):
            read_volume(path, ("labels",))

    def test_h5_fallback_prints_the_dataset(self):
        path = self._path("x.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("labels/original", data=self.bscan)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            read_volume(path, SEGMENTATION_KEYS)
        self.assertIn("labels/original", output.getvalue())

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            read_volume(path, ("labels/original",))
        self.assertEqual(output.getvalue(), "")

    def test_vol(self):
        path = self._path("x.vol")
        _write_vol(path, np.random.default_rng(0).random((3, 5, 8), dtype=np.float32))
        np.testing.assert_array_equal(read_volume(path), read_heyex_vol(path))

    def test_read_bscan_selection(self):
        imageio.imwrite(self._path("a.tif"), self.bscan)
        imageio.imwrite(self._path("s.tif"), np.stack([self.bscan, self.bscan + 1]))
        np.testing.assert_array_equal(read_bscan(self._path("a.tif"), z=7), self.bscan)
        np.testing.assert_array_equal(read_bscan(self._path("s.tif"), z=1), self.bscan + 1)
        for z in (None, 2, -1):
            with self.subTest(z=z), self.assertRaises(ValueError):
                read_bscan(self._path("s.tif"), z=z)


if __name__ == "__main__":
    unittest.main()
