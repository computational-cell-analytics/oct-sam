import os
import tempfile
import unittest

import imageio.v3 as imageio
import nibabel as nib
import numpy as np

from oct_tools.apply_nnunet import _convert_nifti_to_tif


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


if __name__ == "__main__":
    unittest.main()
