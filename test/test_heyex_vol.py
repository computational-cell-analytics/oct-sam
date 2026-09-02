"""Tests for oct_tools.heyex_vol.

The synthetic tests build a VOL file in memory, so they need no external data. The comparison
against the converted HCMS data is skipped when that data set is not available.
"""
import glob
import os
import struct
import tempfile
import unittest

import numpy as np

from oct_tools.heyex_vol import read_heyex_vol, read_vol_header, vol_intensity_transform

FILE_HEADER_SIZE = 2048

# Converted HCMS training data, used for the optional comparison below.
HCMS_H5_DIR = os.path.expanduser("~/Documents/oct-data/pretrain_data/hcms")
HCMS_VOL_DIR = os.path.expanduser(
    "~/Documents/oct-data/public_datasets/hcms_OCT_Manual_Delineations-2018_June_29_b"
    "/OCT_Manual_Delineations-2018_June_29/vol"
)
HCMS_VOLUME = "hc01_spectralis_macula_v1_s1_R"


def _write_vol(
    path,
    bscans,
    size_x_slo=4,
    size_y_slo=4,
    bscan_hdr_size=64,
    scale_x=0.006,
    distance=0.132,
    scale_z=0.0039,
    trailing_bytes=0,
):
    """Write a minimal but valid VOL file holding the given float32 B-scans."""
    num_bscans, size_z, size_x = bscans.shape

    header = bytearray(FILE_HEADER_SIZE)
    header[0:12] = b"HSF-OCT-103\x00"
    struct.pack_into("<iii", header, 12, size_x, num_bscans, size_z)
    struct.pack_into("<ddd", header, 24, scale_x, distance, scale_z)
    struct.pack_into("<ii", header, 48, size_x_slo, size_y_slo)
    struct.pack_into("<dd", header, 56, scale_x, scale_x)
    header[84:88] = b"OD\x00\x00"
    struct.pack_into("<ii", header, 96, 3, bscan_hdr_size)

    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\x7f" * (size_x_slo * size_y_slo))  # SLO image, content is irrelevant here
        for bscan in bscans:
            f.write(b"\x00" * bscan_hdr_size)
            f.write(bscan.astype("<f4").tobytes())
        f.write(b"\x00" * trailing_bytes)


class TestVolHeader(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, "scan.vol")
        self.bscans = np.zeros((3, 5, 7), dtype=np.float32)
        _write_vol(self.path, self.bscans, size_x_slo=8, size_y_slo=6, bscan_hdr_size=128)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_header_fields(self):
        header = read_vol_header(self.path)
        self.assertEqual(header["version"], "HSF-OCT-103")
        self.assertEqual((header["num_bscans"], header["size_z"], header["size_x"]), (3, 5, 7))
        self.assertEqual((header["size_x_slo"], header["size_y_slo"]), (8, 6))
        self.assertEqual(header["bscan_hdr_size"], 128)
        self.assertEqual(header["scan_pattern"], 3)
        self.assertEqual(header["scan_position"], "OD")
        self.assertAlmostEqual(header["scale_z"], 0.0039)
        self.assertAlmostEqual(header["scale_x"], 0.006)
        self.assertAlmostEqual(header["distance"], 0.132)

    def test_truncated_file_raises(self):
        path = os.path.join(self.tmp_dir.name, "short.vol")
        with open(path, "wb") as f:
            f.write(b"\x00" * 100)
        with self.assertRaises(ValueError):
            read_vol_header(path)

    def test_truncated_bscans_raise(self):
        path = os.path.join(self.tmp_dir.name, "cut.vol")
        with open(self.path, "rb") as f:
            content = f.read()
        with open(path, "wb") as f:
            f.write(content[:-40])
        with self.assertRaises(ValueError):
            read_heyex_vol(path)


class TestIntensityTransform(unittest.TestCase):
    def test_matches_the_reference_formula(self):
        data = np.array([[0.0, 1e-3, 0.5, 1.0]], dtype=np.float32)
        expected = np.clip((np.log(data + 2.44e-04) + 8.3) / 8.285, 0, 1)
        expected = np.round(expected * 255).astype(np.uint8)
        np.testing.assert_array_equal(vol_intensity_transform(data.copy()), expected)

    def test_missing_values_become_zero(self):
        data = np.array([[np.finfo(np.float32).max, 1.0]], dtype=np.float32)
        result = vol_intensity_transform(data.copy())
        self.assertEqual(result[0, 0], 0)
        self.assertGreater(result[0, 1], 0)

    def test_output_is_uint8(self):
        data = np.full((2, 3), 0.5, dtype=np.float32)
        self.assertEqual(vol_intensity_transform(data).dtype, np.uint8)


class TestReadHeyexVol(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, "scan.vol")
        rng = np.random.default_rng(0)
        self.bscans = rng.random((4, 6, 9), dtype=np.float32)
        _write_vol(self.path, self.bscans)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_shape_and_dtype(self):
        volume = read_heyex_vol(self.path)
        self.assertEqual(volume.shape, self.bscans.shape)
        self.assertEqual(volume.dtype, np.uint8)

    def test_values_match_the_transformed_input(self):
        expected = vol_intensity_transform(self.bscans.copy())
        np.testing.assert_array_equal(read_heyex_vol(self.path), expected)

    def test_trailing_bytes_are_ignored(self):
        # Heyex exports carry padding after the last B-scan, so the layout comes from the header.
        path = os.path.join(self.tmp_dir.name, "padded.vol")
        _write_vol(path, self.bscans, trailing_bytes=132)
        np.testing.assert_array_equal(read_heyex_vol(path), read_heyex_vol(self.path))

    def test_bscans_are_read_in_order(self):
        bscans = np.stack([np.full((3, 4), v, dtype=np.float32) for v in (0.1, 0.4, 0.9)])
        path = os.path.join(self.tmp_dir.name, "ordered.vol")
        _write_vol(path, bscans)
        volume = read_heyex_vol(path)
        self.assertTrue(volume[0].max() < volume[1].max() < volume[2].max())


@unittest.skipUnless(
    os.path.isdir(HCMS_H5_DIR) and os.path.isdir(HCMS_VOL_DIR),
    "HCMS data set is not available.",
)
class TestAgainstConvertedHcmsData(unittest.TestCase):
    """The reader must reproduce the data that eyepy produced for the training set."""

    def test_matches_converted_h5(self):
        import h5py

        volume = read_heyex_vol(os.path.join(HCMS_VOL_DIR, f"{HCMS_VOLUME}.vol"))
        references = sorted(glob.glob(os.path.join(HCMS_H5_DIR, f"{HCMS_VOLUME}_*.h5")))
        self.assertEqual(len(references), volume.shape[0])
        for z, reference in enumerate(references):
            with h5py.File(reference, "r") as f:
                np.testing.assert_array_equal(f["image"][:], volume[z], err_msg=f"B-scan {z}")


if __name__ == "__main__":
    unittest.main()
