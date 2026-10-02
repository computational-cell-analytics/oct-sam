import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import h5py
import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from oct_tools.apply_nnunet import (
    _convert_nifti_to_tif, _convert_to_nnunet_format, _create_dataset, retrain_model_nnunet,
)


class TestConvertNiftiToTif(unittest.TestCase):
    def test_label_dtype(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            label = np.zeros((8, 16), dtype=np.uint8)
            label[2:5] = 1
            nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(tmp_dir, "seg.nii.gz"))

            for label_data, dtype in ((False, np.uint8), (True, np.uint32)):
                out_dir = os.path.join(tmp_dir, str(label_data))
                _convert_nifti_to_tif(tmp_dir, out_dir, label_data=label_data)
                tif = imageio.imread(os.path.join(out_dir, "seg.tif"))
                self.assertEqual(tif.dtype, dtype)
                np.testing.assert_array_equal(tif, label)


class TestConvertToNnunetFormat(unittest.TestCase):
    def test_rgb_and_stack(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(input_dir)
            gray = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
            imageio.imwrite(os.path.join(input_dir, "rgb.tif"), np.stack([gray] * 3, axis=-1))
            imageio.imwrite(os.path.join(input_dir, "stack.tif"), np.stack([gray, gray]))

            _convert_to_nnunet_format(input_dir, output_dir)

            self.assertEqual(
                sorted(os.listdir(output_dir)),
                ["rgb_0000.nii.gz", "stack_z000_0000.nii.gz", "stack_z001_0000.nii.gz"],
            )
            rgb = nib.load(os.path.join(output_dir, "rgb_0000.nii.gz"))
            self.assertEqual(rgb.get_data_dtype(), np.uint8)
            np.testing.assert_array_equal(rgb.get_fdata(), gray)


class TestCreateDataset(unittest.TestCase):
    BINARY = {"channel_names": {"0": "OCT"}, "labels": {"background": 0, "retina": 1}}
    LAYERS = {"channel_names": {"0": "intensity"}, "labels": {"background": 0, **{f"L{i}": i for i in range(1, 8)}}}

    def _write(self, folder, name, data):
        os.makedirs(folder, exist_ok=True)
        imageio.imwrite(os.path.join(folder, name), data)

    def test_binary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, "images")
            label_dir = os.path.join(tmp_dir, "labels")
            dataset_dir = os.path.join(tmp_dir, "dataset")
            gray = np.arange(8 * 16, dtype=np.uint8).reshape(8, 16)
            for name, data in (("a.tif", gray), ("b.tif", gray), ("s.tif", np.stack([gray, gray]))):
                self._write(image_dir, name, data)
            label = np.zeros((8, 16), dtype=np.uint32)
            label[2:4] = 1
            label[4:6] = 2
            self._write(label_dir, "a.tif", label)
            self._write(label_dir, "s_z001.tif", label)

            self.assertEqual(_create_dataset(image_dir, label_dir, dataset_dir, self.BINARY, "edit_v3"), 2)

            self.assertEqual(sorted(os.listdir(os.path.join(dataset_dir, "imagesTr"))),
                             ["a_0000.nii.gz", "s_z001_0000.nii.gz"])
            self.assertEqual(sorted(os.listdir(os.path.join(dataset_dir, "labelsTr"))),
                             ["a.nii.gz", "s_z001.nii.gz"])
            nifti = nib.load(os.path.join(dataset_dir, "labelsTr", "a.nii.gz"))
            self.assertEqual(nifti.get_data_dtype(), np.uint8)
            np.testing.assert_array_equal(np.asanyarray(nifti.dataobj), label > 0)
            with open(os.path.join(dataset_dir, "dataset.json")) as f:
                dataset_json = json.load(f)
            self.assertEqual(dataset_json["numTraining"], 2)
            self.assertEqual(dataset_json["labels"], self.BINARY["labels"])

    def test_layers_from_h5(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            h5_dir = os.path.join(tmp_dir, "h5")
            dataset_dir = os.path.join(tmp_dir, "dataset")
            os.makedirs(h5_dir)
            image = np.full((8, 16), 50, dtype=np.uint8)
            label = np.repeat(np.arange(8, dtype=np.uint8), 2)[None].repeat(8, axis=0)
            with h5py.File(os.path.join(h5_dir, "x.h5"), "w") as f:
                f["image"] = image
                f["labels/edit_v3"] = label

            _create_dataset(h5_dir, h5_dir, dataset_dir, self.LAYERS, "edit_v3")

            nifti = nib.load(os.path.join(dataset_dir, "labelsTr", "x.nii.gz"))
            np.testing.assert_array_equal(np.asanyarray(nifti.dataobj), label)
            with open(os.path.join(dataset_dir, "dataset.json")) as f:
                self.assertEqual(json.load(f)["labels"], self.LAYERS["labels"])

    def test_unexpected_label_id(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, "images")
            label_dir = os.path.join(tmp_dir, "labels")
            self._write(image_dir, "a.tif", np.zeros((8, 16), dtype=np.uint8))
            self._write(label_dir, "a.tif", np.full((8, 16), 9, dtype=np.uint8))
            with self.assertRaises(ValueError):
                _create_dataset(image_dir, label_dir, os.path.join(tmp_dir, "dataset"), self.LAYERS, "edit_v3")

    def test_label_without_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = os.path.join(tmp_dir, "images")
            label_dir = os.path.join(tmp_dir, "labels")
            self._write(image_dir, "a.tif", np.zeros((8, 16), dtype=np.uint8))
            self._write(label_dir, "b.tif", np.ones((8, 16), dtype=np.uint8))
            with self.assertRaises(ValueError):
                _create_dataset(image_dir, label_dir, os.path.join(tmp_dir, "dataset"), self.BINARY, "edit_v3")


class TestRetrainExistingDataset(unittest.TestCase):
    def test_warning(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_vars = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
            env = {name: os.path.join(tmp_dir, name) for name in env_vars}
            existing = os.path.join(env["nnUNet_raw"], "Dataset014_OCT-2d-retrain-013")
            os.makedirs(existing)
            output = io.StringIO()
            with mock.patch.dict(os.environ, env), mock.patch("subprocess.run") as run, \
                    contextlib.redirect_stdout(output):
                retrain_model_nnunet(tmp_dir, tmp_dir, pretrained_id="013", dataset_id="014")
            run.assert_not_called()
            self.assertIn("Warning", output.getvalue())
            self.assertIn(existing, output.getvalue())


if __name__ == "__main__":
    unittest.main()
