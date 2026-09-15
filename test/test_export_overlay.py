import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import tifffile

from oct_tools.export_overlay import export_overlay
from oct_tools.layer_information import LAYER_COLORS_CUSTOM

DEPTH, HEIGHT, WIDTH = 3, 40, 60


def _make_data():
    image = np.linspace(0, 255, DEPTH * HEIGHT * WIDTH, dtype=np.float32)
    image = image.reshape(DEPTH, HEIGHT, WIDTH).astype(np.uint8)
    seg = np.zeros((DEPTH, HEIGHT, WIDTH), dtype=np.uint32)
    seg[:, 10:20, :] = 1
    seg[:, 20:30, :] = 7
    return image, seg


class TestExportOverlay(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.image, self.seg = _make_data()
        self.image_path = os.path.join(self.tmp_dir.name, "image.tif")
        self.seg_path = os.path.join(self.tmp_dir.name, "seg.tif")
        imageio.imwrite(self.image_path, self.image)
        imageio.imwrite(self.seg_path, self.seg)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _export(self, name, **kwargs):
        output_path = os.path.join(self.tmp_dir.name, name)
        export_overlay(self.image_path, self.seg_path, output_path, **kwargs)
        return output_path

    def test_composite_holds_both_channels(self):
        stack = tifffile.imread(self._export("overlay.tif", mode="composite"))
        self.assertEqual(stack.shape, (DEPTH, 2, HEIGHT, WIDTH))
        np.testing.assert_array_equal(stack[:, 0], self.image)
        np.testing.assert_array_equal(stack[:, 1], self.seg.astype(np.uint8))

    def test_rgb_uses_the_layer_colors(self):
        rgb = imageio.imread(self._export("overlay_rgb.tif", mode="rgb"))
        self.assertEqual(rgb.shape, (DEPTH, HEIGHT, WIDTH, 3))
        self.assertEqual(rgb.dtype, np.uint8)

        # The background keeps the grayscale value, a layer is tinted towards its own color.
        np.testing.assert_array_equal(rgb[0, 0, :, 0], rgb[0, 0, :, 2])
        rpe = LAYER_COLORS_CUSTOM[7][:3]
        self.assertEqual(int(np.argmax(rgb[0, 25, 30])), int(np.argmax(rpe)))

    def test_slices_are_selected(self):
        # ImageJ drops a z-axis of length one on read-back.
        stack = tifffile.imread(self._export("one.tif", mode="composite", slices=[1]))
        self.assertEqual(stack.shape, (2, HEIGHT, WIDTH))
        np.testing.assert_array_equal(stack[0], self.image[1])
        np.testing.assert_array_equal(stack[1], self.seg[1].astype(np.uint8))

    def test_png_needs_a_single_slice(self):
        with self.assertRaises(ValueError):
            self._export("overlay.png", mode="rgb")
        png = imageio.imread(self._export("overlay.png", mode="rgb", slices=[2]))
        self.assertEqual(png.shape, (HEIGHT, WIDTH, 3))

    def test_shape_mismatch_raises(self):
        imageio.imwrite(self.seg_path, self.seg[:2])
        with self.assertRaises(ValueError):
            self._export("overlay.tif")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self._export("overlay.tif", mode="nope")

    def test_composite_refuses_a_non_tif_path(self):
        # tifffile ignores the extension, so a .png would hold an unopenable TIF.
        with self.assertRaises(ValueError):
            self._export("overlay.png")

    def test_h5_segmentation_accepts_the_seg_key(self):
        import h5py
        self.seg_path = os.path.join(self.tmp_dir.name, "seg.h5")
        with h5py.File(self.seg_path, "w") as f:
            f.create_dataset("seg", data=self.seg)
        stack = tifffile.imread(self._export("from_h5.tif", mode="composite"))
        np.testing.assert_array_equal(stack[:, 1], self.seg.astype(np.uint8))

    def test_label_ids_above_255_are_refused(self):
        seg = self.seg.copy()
        seg[:, 10:12, :] = 300
        imageio.imwrite(self.seg_path, seg)
        with self.assertRaises(ValueError):
            self._export("overlay.tif")

    def test_warning_ids_are_not_black(self):
        seg = np.zeros_like(self.seg)
        seg[:, 10:20, :] = 42  # not a valid layer ID
        imageio.imwrite(self.seg_path, seg)
        rgb = imageio.imread(self._export("warn.tif", mode="rgb"))
        self.assertGreater(int(rgb[0, 15, 30].max()), int(rgb[0, 0, 30].max()))

    def test_rgb_image_is_averaged(self):
        rgb_image = np.repeat(self.image[..., np.newaxis], 3, axis=-1)
        self.image_path = os.path.join(self.tmp_dir.name, "rgb_image.tif")
        imageio.imwrite(self.image_path, rgb_image[0])
        imageio.imwrite(self.seg_path, self.seg[0])
        stack = tifffile.imread(self._export("from_rgb.tif", mode="composite"))
        self.assertEqual(stack.shape, (2, HEIGHT, WIDTH))
        np.testing.assert_array_equal(stack[0], self.image[0])


if __name__ == "__main__":
    unittest.main()
