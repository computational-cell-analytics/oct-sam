import os
import tempfile
import unittest
from unittest import mock

import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from oct_tools.segmentation.apply_nnunet import _convert_nifti_to_tif, _convert_to_nnunet_format, apply_model_nnunet


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

    def test_ignores_other_extensions(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(input_dir)
            for name in ("a.tif", "b.tif"):
                imageio.imwrite(os.path.join(input_dir, name), np.zeros((8, 16), dtype=np.uint8))
            with open(os.path.join(input_dir, "notes.tif.txt"), "w") as f:
                f.write("not an image")

            _convert_to_nnunet_format(input_dir, output_dir)

            self.assertEqual(sorted(os.listdir(output_dir)), ["a_0000.nii.gz", "b_0000.nii.gz"])


def _fake_predict(args, check):
    """Write one label NIfTI for each nnU-Net input image, as nnUNetv2_predict does."""
    input_dir, output_dir = args[args.index("-i") + 1], args[args.index("-o") + 1]
    for name in os.listdir(input_dir):
        image = np.asanyarray(nib.load(os.path.join(input_dir, name)).dataobj)
        label = (image > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(output_dir, name.replace("_0000.nii.gz", ".nii.gz")))


class TestApplySingleFile(unittest.TestCase):
    def test_file_name_and_stack(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.zeros((8, 16), dtype=np.uint8)
            image[2:5] = 200
            for data in (image, np.stack([image] * 3)):
                input_path = os.path.join(tmp_dir, "my scan.v1_0000.TIFF")
                output_path = os.path.join(tmp_dir, "out", "sub", "seg.tif")
                imageio.imwrite(input_path, data)
                with mock.patch("subprocess.run", side_effect=_fake_predict):
                    apply_model_nnunet(input_path, output_path)
                seg = imageio.imread(output_path)
                self.assertEqual(seg.dtype, np.uint32)
                np.testing.assert_array_equal(seg, data > 0)

    def test_output_must_be_tif(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "a.tif")
            imageio.imwrite(input_path, np.zeros((8, 16), dtype=np.uint8))
            with mock.patch("subprocess.run") as run, self.assertRaises(ValueError):
                apply_model_nnunet(input_path, os.path.join(tmp_dir, "out"))
            run.assert_not_called()

    def test_output_must_not_be_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image = np.full((8, 16), 7, dtype=np.uint8)
            input_path = os.path.join(tmp_dir, "a.tif")
            imageio.imwrite(input_path, image)
            for path in (input_path, tmp_dir, tmp_dir + "/"):
                with self.subTest(path=path), mock.patch("subprocess.run") as run, self.assertRaises(ValueError):
                    apply_model_nnunet(path, path)
                run.assert_not_called()
            np.testing.assert_array_equal(imageio.imread(input_path), image)


if __name__ == "__main__":
    unittest.main()
