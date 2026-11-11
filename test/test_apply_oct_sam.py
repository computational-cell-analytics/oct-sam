import os
import tempfile
import types
import unittest
from unittest import mock

import h5py
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

    def test_force_overwrite_segments_existing_output(self):
        os.makedirs(self.out_dir)
        open(os.path.join(self.out_dir, "scan_1.tif"), "w").close()
        calls = self._run(np.zeros((2, 8, 10), dtype=np.uint8), force_overwrite=True)
        self.assertEqual([name for _, name in calls], ["scan_0.tif", "scan_1.tif"])

    def _write_h5(self, **datasets):
        path = os.path.join(self.tmp_dir.name, "scan.h5")
        with h5py.File(path, "w") as f:
            for key, data in datasets.items():
                f.create_dataset(key, data=data)
        return path

    def test_h5_input_in_output_folder(self):
        path = self._write_h5(image=np.zeros((8, 10), dtype=np.uint8))
        apply_oct_sam.apply_model_sam_2d(self.tmp_dir.name, "model.pt", self.tmp_dir.name, output_extension="h5")
        self.assertEqual([c.args[3] for c in self.segment.call_args_list], [path])

    def test_h5_with_segmentation_is_skipped(self):
        image = np.zeros((8, 10), dtype=np.uint8)
        self._write_h5(image=image, segmentation=image.astype(np.uint32))
        apply_oct_sam.apply_model_sam_2d(self.tmp_dir.name, "model.pt", self.tmp_dir.name, output_extension="h5")
        self.segment.assert_not_called()

    def test_tif_output_must_not_replace_input(self):
        image = np.full((8, 10), 7, dtype=np.uint8)
        path = os.path.join(self.tmp_dir.name, "scan.tif")
        imageio.imwrite(path, image)
        for input_path in (self.tmp_dir.name, path):
            with self.subTest(input_path=input_path), self.assertRaises(ValueError):
                apply_oct_sam.apply_model_sam_2d(input_path, "model.pt", self.tmp_dir.name, force_overwrite=True)
        self.segment.assert_not_called()
        np.testing.assert_array_equal(imageio.imread(path), image)

    def test_stack_output_must_not_replace_other_input(self):
        imageio.imwrite(os.path.join(self.tmp_dir.name, "scan.tif"), np.zeros((2, 8, 10), dtype=np.uint8))
        self._write_h5(image=np.zeros((8, 10), dtype=np.uint8))
        os.rename(os.path.join(self.tmp_dir.name, "scan.h5"), os.path.join(self.tmp_dir.name, "scan_0.h5"))
        with self.assertRaises(ValueError):
            apply_oct_sam.apply_model_sam_2d(self.tmp_dir.name, "model.pt", self.tmp_dir.name, output_extension="h5")


class TestSegmentImage(unittest.TestCase):
    """The model is mocked. The tests check the H5 output of _segment_image."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.path = os.path.join(self.tmp_dir.name, "scan.h5")
        self.image = np.zeros((8, 10), dtype=np.uint8)
        self.segmenter = types.SimpleNamespace(
            initialize=lambda image, verbose: None,
            _foreground=None, _boundary_distances=None, _center_distances=None,
        )
        self.seg = np.ones((8, 10), dtype=np.uint32)
        self.prompts = np.array([[2.0, 3.0], [5.0, 6.0]])
        patches = [
            mock.patch.object(apply_oct_sam, "_derive_prompts_sam", side_effect=lambda *args: self.prompts),
            mock.patch.object(apply_oct_sam, "_segment_from_prompts", side_effect=lambda *args, **kwargs: self.seg),
            mock.patch.object(apply_oct_sam, "watershed_from_center_and_boundary_distances",
                              side_effect=lambda *args: self.seg),
            mock.patch.object(apply_oct_sam, "postprocess_segmentation", side_effect=lambda seg, *args: seg),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _segment(self, use_prompts=True):
        apply_oct_sam._segment_image(None, self.segmenter, self.image, self.path, use_prompts=use_prompts)

    def test_h5_output_is_replaced(self):
        self._segment()
        self.seg = np.full((8, 10), 2, dtype=np.uint32)
        self.prompts = np.array([[1.0, 1.0]])
        self._segment()
        with h5py.File(self.path, "r") as f:
            np.testing.assert_array_equal(f["segmentation"][:], self.seg)
            np.testing.assert_array_equal(f["prompts"][:], self.prompts)

    def test_h5_without_prompts(self):
        self._segment()
        self.seg = np.full((8, 10), 2, dtype=np.uint32)
        self._segment(use_prompts=False)
        with h5py.File(self.path, "r") as f:
            np.testing.assert_array_equal(f["segmentation"][:], self.seg)
            self.assertNotIn("prompts", f)

    def test_h5_keeps_other_datasets(self):
        with h5py.File(self.path, "w") as f:
            f.create_dataset("image", data=self.image)
        self._segment()
        with h5py.File(self.path, "r") as f:
            self.assertEqual(set(f.keys()), {"image", "segmentation", "prompts"})


if __name__ == "__main__":
    unittest.main()
