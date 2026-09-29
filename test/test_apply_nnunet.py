import os
import tempfile
import unittest

import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from oct_tools.apply_nnunet import _convert_nifti_to_tif, _convert_to_nnunet_format


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


if __name__ == "__main__":
    unittest.main()
