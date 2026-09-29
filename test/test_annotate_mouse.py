import importlib.util
import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "mouse", "annotate.py")
_spec = importlib.util.spec_from_file_location("annotate", SCRIPT)
annotate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(annotate)


class TestCollectBscans(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.image_dir, self.label_dir, self.slice_dir = (
            os.path.join(self.tmp_dir.name, d) for d in ("images", "labels", "slices")
        )
        for d in (self.image_dir, self.label_dir, self.slice_dir):
            os.makedirs(d)
        bscan = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
        self.label = np.ones((8, 16), dtype=np.uint32)
        imageio.imwrite(os.path.join(self.image_dir, "rgb.tif"), np.stack([bscan] * 3, axis=-1))
        imageio.imwrite(os.path.join(self.image_dir, "stack.tif"), np.stack([bscan, bscan]))
        imageio.imwrite(os.path.join(self.image_dir, "labelstack.tif"), np.stack([bscan, bscan]))
        imageio.imwrite(os.path.join(self.label_dir, "rgb.tif"), self.label)
        imageio.imwrite(os.path.join(self.label_dir, "stack_z000.tif"), self.label)
        imageio.imwrite(os.path.join(self.label_dir, "stack_z001.tif"), self.label)
        imageio.imwrite(os.path.join(self.label_dir, "labelstack.tif"), np.stack([self.label, self.label]))

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_2d_and_stacks(self):
        images, labels = annotate.collect_bscans(self.image_dir, self.label_dir, self.slice_dir)

        self.assertEqual(
            [os.path.basename(p) for p in images],
            ["labelstack_z000.tif", "labelstack_z001.tif", "rgb.tif", "stack_z000.tif", "stack_z001.tif"],
        )
        self.assertEqual(images[2], os.path.join(self.image_dir, "rgb.tif"))
        self.assertEqual(imageio.imread(images[0]).shape, (8, 16))
        np.testing.assert_array_equal(labels[0], self.label)
        self.assertEqual(labels[3], os.path.join(self.label_dir, "stack_z000.tif"))

    def test_missing_label(self):
        os.remove(os.path.join(self.label_dir, "stack_z001.tif"))
        with self.assertRaises(FileNotFoundError):
            annotate.collect_bscans(self.image_dir, self.label_dir, self.slice_dir)


if __name__ == "__main__":
    unittest.main()
