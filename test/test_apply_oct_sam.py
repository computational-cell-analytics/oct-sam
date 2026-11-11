import os
import tempfile
import unittest
from unittest import mock

import imageio.v3 as imageio
import numpy as np

from oct_tools import apply_oct_sam


class TestApplyModelSam2d(unittest.TestCase):
    """The model is mocked. The tests check which B-scans are segmented and where the output goes."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.out_dir = os.path.join(self.tmp_dir.name, "out")
        patches = [
            mock.patch.object(apply_oct_sam, "get_predictor_and_decoder", return_value=(None, None)),
            mock.patch.object(apply_oct_sam, "get_amg", return_value=None),
            mock.patch.object(apply_oct_sam, "_segment_image", return_value=(None, None)),
        ]
        self.segment = [p.start() for p in patches][-1]
        for p in patches:
            self.addCleanup(p.stop)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _run(self, image, **kwargs):
        path = os.path.join(self.tmp_dir.name, "scan.tif")
        imageio.imwrite(path, image)
        apply_oct_sam.apply_model_sam_2d(path, "model.pt", self.out_dir, **kwargs)
        return [(c.args[2], os.path.basename(c.args[3])) for c in self.segment.call_args_list]

    def test_stack_with_slices(self):
        stack = np.stack([np.full((8, 10), z, dtype=np.uint8) for z in range(3)])
        calls = self._run(stack, slices=[0, 2])
        self.assertEqual([name for _, name in calls], ["scan_0.tif", "scan_2.tif"])
        np.testing.assert_array_equal(calls[1][0], stack[2])

    def test_stack_without_slices_segments_all(self):
        calls = self._run(np.zeros((3, 8, 10), dtype=np.uint8))
        self.assertEqual([name for _, name in calls], ["scan_0.tif", "scan_1.tif", "scan_2.tif"])

    def test_invalid_slices_raise(self):
        for slices in ([3], [-1]):
            with self.subTest(slices=slices), self.assertRaises(ValueError):
                self._run(np.zeros((3, 8, 10), dtype=np.uint8), slices=slices)
        self.segment.assert_not_called()

    def test_rgb_bscan_is_one_image(self):
        calls = self._run(np.zeros((8, 10, 3), dtype=np.uint8), slices=[1])
        self.assertEqual([name for _, name in calls], ["scan.tif"])
        self.assertEqual(calls[0][0].shape, (8, 10))

    def test_existing_output_is_skipped(self):
        os.makedirs(self.out_dir)
        open(os.path.join(self.out_dir, "scan_1.tif"), "w").close()
        calls = self._run(np.zeros((2, 8, 10), dtype=np.uint8))
        self.assertEqual([name for _, name in calls], ["scan_0.tif"])


if __name__ == "__main__":
    unittest.main()
