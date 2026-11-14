import json
import os
import tempfile
import unittest

import h5py
import nibabel as nib
import numpy as np
from scipy.io import savemat

from oct_tools.data_processing import (
    Sample, _load_duke_data, _surfaces_to_label, case_source, copy_cases, create_train_val_splits_for_finetuning,
    legacy_nnunet_name, merge_photoreceptor_segments, prepare_dataset, read_duke_dme, read_umg_rp,
    split_nnunet_dataset, split_sam_data, split_semantic_data, standard_name, umg_rp_source_name,
)

SPLIT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "doc", "train_splits")


def _write_split(path, train, val):
    with open(path, "w") as f:
        json.dump([{"train": train, "val": val}], f)


def _write_nnunet_case(folder, name, image, label):
    os.makedirs(os.path.join(folder, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(folder, "labelsTr"), exist_ok=True)
    nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(folder, "imagesTr", f"{name}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(label, np.eye(4)), os.path.join(folder, "labelsTr", f"{name}.nii.gz"))


class TestSurfacesToLabel(unittest.TestCase):
    def test_bands(self):
        control_pts = np.empty((1, 4), dtype=object)
        # 1-based (x, y) points of flat surfaces at y = 2, 5 and 8. Surface 1 has no points.
        control_pts[0, 0] = np.array([[1, 2], [5, 2]])
        control_pts[0, 1] = np.zeros((0, 2))
        control_pts[0, 2] = np.array([[1, 5], [5, 5]])
        control_pts[0, 3] = np.array([[1, 8], [5, 8]])

        label = _surfaces_to_label(control_pts, (1, 10, 5))

        expected = np.zeros((10, 5), dtype=np.uint8)
        expected[1:4] = 1
        expected[4:7] = 2
        np.testing.assert_array_equal(label[0], expected)


class TestDukeDme(unittest.TestCase):
    @staticmethod
    def _volume():
        images = np.arange(10 * 6 * 3, dtype=np.uint8).reshape(10, 6, 3)
        layers = np.full((3, 6, 3), np.nan)
        # 1-based surfaces at y = 2, 5 and 8 in the columns 1-4 of B-scan 1.
        for s, y in enumerate((2, 5, 8)):
            layers[s, 1:5, 1] = y
        return images, layers

    def test_load_duke_data(self):
        images, layers = self._volume()
        images_zyx, labels = _load_duke_data({"images": images, "manualLayers1": layers})

        np.testing.assert_array_equal(images_zyx[1], images[:, :, 1])
        self.assertEqual(labels[0].max(), 0)
        self.assertEqual(labels[2].max(), 0)
        expected = np.zeros((10, 6), dtype=np.uint8)
        expected[1:5, 1:5] = 1
        expected[4:8, 1:5] = 2
        np.testing.assert_array_equal(labels[1], expected)

    def test_read_duke_dme(self):
        images, layers = self._volume()
        with tempfile.TemporaryDirectory() as tmp_dir:
            savemat(os.path.join(tmp_dir, "Subject_03.mat"), {"images": images, "manualLayers1": layers})
            samples = list(read_duke_dme(tmp_dir))

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][:3], ("duke_03_z01", "oct_20003001", "Subject_03_z000"))
        np.testing.assert_array_equal(samples[0].image, images[:, 1:5, 1])
        self.assertEqual(samples[0].label.shape, (10, 4))
        self.assertTrue((samples[0].label[1:8] > 0).all())


class TestLabelsAndNames(unittest.TestCase):
    def test_merge_photoreceptor_segments(self):
        label = np.arange(9, dtype=np.uint8)
        merged = merge_photoreceptor_segments(label)
        np.testing.assert_array_equal(merged, [0, 1, 2, 3, 4, 5, 6, 6, 7])
        np.testing.assert_array_equal(label, np.arange(9))
        with self.assertRaises(ValueError):
            merge_photoreceptor_segments(np.arange(8))

    # Standard name, legacy nnU-Net name, legacy H5 name (None if it cannot be converted).
    NAMES = (
        ("hcms_hc_01_z00", "oct_11001_000", "hc01_spectralis_macula_v1_s1_R_000"),
        ("hcms_ms_21_z48", "oct_12021_048", "ms21_spectralis_macula_v1_s1_R_048"),
        ("duke_01_z10", "oct_20001010", None),
        ("duke_10_z51", "oct_20010051", None),
        ("umgrp_012_2023_OD_z08", "oct_30012_2023_OD_008", "RP012_2023_OD_z08"),
        ("umgrp_101_2025_OS_z25", "oct_30101_2025_OS_025", "RP101_2025_OS_z25"),
    )

    def test_standard_name(self):
        for standard, legacy_nnunet, legacy_h5 in self.NAMES:
            self.assertEqual(standard_name(standard), standard)
            self.assertEqual(standard_name(legacy_nnunet), standard)
            self.assertEqual(legacy_nnunet_name(standard), legacy_nnunet)
            if legacy_h5 is not None:
                self.assertEqual(standard_name(legacy_h5), standard)
        self.assertEqual(standard_name("oct_20001_010"), "duke_01_z10")
        for name in ("Subject_01_z000", "RP01_05_07_2023OS025", "oct_40001_000", "hcms_xx_01_z00"):
            with self.assertRaises(ValueError):
                standard_name(name)

    def test_umg_rp_source_name(self):
        self.assertEqual(umg_rp_source_name("oct_30012_2023_OD_008"), "RP012_2023_OD_z08")
        self.assertEqual(umg_rp_source_name("umgrp_012_2023_OD_z08"), "RP012_2023_OD_z08")
        with self.assertRaises(ValueError):
            umg_rp_source_name("oct_11001_000")

    def test_case_source(self):
        self.assertEqual(case_source("oct_11001_000"), "hcms")
        self.assertEqual(case_source("hcms_ms_21_z48"), "hcms")
        self.assertEqual(case_source("oct_20010030"), "duke_dme")
        self.assertEqual(case_source("duke_10_z30"), "duke_dme")
        self.assertEqual(case_source("oct_30012_2023_OD_008"), "umg_rp")
        self.assertEqual(case_source("RP012_2023_OD_z08"), "umg_rp")
        with self.assertRaises(ValueError):
            case_source("oct_40001_000")


class TestReadUmgRp(unittest.TestCase):
    def test_skips(self):
        label = np.zeros((8, 16), dtype=np.int64)
        label[2:4] = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            for name, key in (
                ("RP012_2023_OD_z08", "labels/edit_v3"),
                ("RP122_2025_OS_z08", "labels/edit_v3"),
                ("RP013_2023_OS_z10", "labels/original"),
            ):
                with h5py.File(os.path.join(tmp_dir, f"{name}.h5"), "w") as f:
                    f.create_dataset("image", data=np.full((8, 16), 3.0))
                    f.create_dataset(key, data=label)
            samples = list(read_umg_rp(tmp_dir))

        self.assertEqual([s[:3] for s in samples],
                         [("umgrp_012_2023_OD_z08", "oct_30012_2023_OD_008", "RP012_2023_OD_z08")])
        self.assertEqual(samples[0].image.dtype, np.uint8)
        self.assertEqual(samples[0].label.dtype, np.uint8)
        np.testing.assert_array_equal(samples[0].label, label)


class TestPrepareDataset(unittest.TestCase):
    @staticmethod
    def _samples(value=5):
        label = np.zeros((8, 16), dtype=np.uint8)
        label[2:4] = 3
        image = np.full((8, 16), value, dtype=np.uint8)
        return [Sample(f"hcms_hc_01_z0{z}", f"oct_11001_00{z}", f"hc01_00{z}", image, label) for z in (0, 1)]

    def test_nnunet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            count = prepare_dataset(self._samples(), tmp_dir, "nnunet", (3.87, 5.8), binary=True)

            self.assertEqual(count, 2)
            self.assertEqual(sorted(os.listdir(os.path.join(tmp_dir, "imagesTr"))),
                             ["hcms_hc_01_z00_0000.nii.gz", "hcms_hc_01_z01_0000.nii.gz"])
            self.assertEqual(sorted(os.listdir(os.path.join(tmp_dir, "labelsTr"))),
                             ["hcms_hc_01_z00.nii.gz", "hcms_hc_01_z01.nii.gz"])
            label = nib.load(os.path.join(tmp_dir, "labelsTr", "hcms_hc_01_z00.nii.gz"))
            np.testing.assert_allclose(np.diag(label.affine)[:2], (3.87, 5.8))
            self.assertEqual(sorted(np.unique(np.asanyarray(label.dataobj)).tolist()), [0, 1])

    def test_legacy_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepare_dataset(self._samples(), os.path.join(tmp_dir, "nnunet"), "nnunet", (3.87, 5.8), legacy_names=True)
            prepare_dataset(self._samples(), os.path.join(tmp_dir, "h5"), "h5", (3.87, 5.8), legacy_names=True)

            self.assertEqual(sorted(os.listdir(os.path.join(tmp_dir, "nnunet", "labelsTr"))),
                             ["oct_11001_000.nii.gz", "oct_11001_001.nii.gz"])
            self.assertEqual(sorted(os.listdir(os.path.join(tmp_dir, "h5"))), ["hc01_000.h5", "hc01_001.h5"])

    def test_h5_rerun(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepare_dataset(self._samples(5), tmp_dir, "h5", (3.87, 5.8))
            prepare_dataset(self._samples(9), tmp_dir, "h5", (3.87, 5.8))

            self.assertEqual(sorted(os.listdir(tmp_dir)), ["hcms_hc_01_z00.h5", "hcms_hc_01_z01.h5"])
            with h5py.File(os.path.join(tmp_dir, "hcms_hc_01_z00.h5"), "r") as f:
                self.assertEqual(sorted(f.keys()), ["image", "masks"])
                self.assertEqual(f["image"][0, 0], 9)
                self.assertEqual(f["masks"][:].max(), 3)

    def test_unknown_format(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                prepare_dataset(self._samples(), tmp_dir, "tif", (3.87, 5.8))


class TestSplits(unittest.TestCase):
    def test_copy_cases_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            os.makedirs(input_dir)
            for name in ("oct_11001_000_0000.nii.gz", "oct_11001_000.nii.gz", "oct_11001_0001.nii.gz",
                         "oct_11001_001.nii.gz", "RP012_2023_OD_z08.h5", "RP012_2023_OD_z080.h5"):
                open(os.path.join(input_dir, name), "w").close()

            output_dir = os.path.join(tmp_dir, "output")
            missing = copy_cases(["oct_11001_000", "RP012_2023_OD_z08", "oct_11001_002"], input_dir, output_dir)

            self.assertEqual(missing, ["oct_11001_002"])
            self.assertEqual(sorted(os.listdir(output_dir)),
                             ["RP012_2023_OD_z08.h5", "oct_11001_000.nii.gz", "oct_11001_000_0000.nii.gz"])

    def test_split_sam_data(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            os.makedirs(input_dir)
            for name in ("RP001_2023_OS_z25", "RP002_2023_OS_z25", "RP003_2023_OS_z25"):
                open(os.path.join(input_dir, f"{name}.h5"), "w").close()
            json_file = os.path.join(tmp_dir, "train_splits_sam_n002.json")
            _write_split(json_file, ["RP001_2023_OS_z25", "RP002_2023_OS_z25"], ["RP003_2023_OS_z25"])

            missing = split_sam_data(input_dir, os.path.join(tmp_dir, "out"), json_file)

            self.assertEqual(missing, [])
            self.assertEqual(sorted(os.listdir(os.path.join(tmp_dir, "out", "train_n002"))),
                             ["RP001_2023_OS_z25.h5", "RP002_2023_OS_z25.h5"])
            self.assertEqual(os.listdir(os.path.join(tmp_dir, "out", "val")), ["RP003_2023_OS_z25.h5"])

    def test_copy_cases_translates_names(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            os.makedirs(input_dir)
            names = ("duke_01_z10_0000.nii.gz", "duke_01_z10.nii.gz", "oct_11001_000.nii.gz", "RP012_2023_OD_z08.h5")
            for name in names:
                open(os.path.join(input_dir, name), "w").close()

            output_dir = os.path.join(tmp_dir, "output")
            missing = copy_cases(
                ["oct_20001010", "hcms_hc_01_z00", "umgrp_012_2023_OD_z08", "duke_02_z10"], input_dir, output_dir
            )

            self.assertEqual(missing, ["duke_02_z10"])
            self.assertEqual(sorted(os.listdir(output_dir)), [
                "RP012_2023_OD_z08.h5", "duke_01_z10.nii.gz", "duke_01_z10_0000.nii.gz", "oct_11001_000.nii.gz"
            ])

    def test_split_nnunet_dataset(self):
        image = np.zeros((8, 16), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = os.path.join(tmp_dir, "input")
            for name in ("hcms_hc_01_z00", "hcms_hc_01_z01", "duke_01_z10", "duke_01_z15"):
                _write_nnunet_case(input_dir, name, image, image)
            json_file = os.path.join(tmp_dir, "split.json")
            with open(json_file, "w") as f:
                json.dump([
                    {"train": ["oct_11001_000", "oct_20001010"], "val": ["oct_11001_001"]},
                    {"train": ["oct_11001_001", "oct_20001015"], "val": ["oct_11001_000", "oct_20001020"]},
                ], f)

            output_dir = os.path.join(tmp_dir, "output")
            missing = split_nnunet_dataset(input_dir, output_dir, json_file)

            self.assertEqual(missing, [])
            self.assertEqual(sorted(os.listdir(os.path.join(output_dir, "labelsTr"))),
                             ["duke_01_z10.nii.gz", "hcms_hc_01_z00.nii.gz", "hcms_hc_01_z01.nii.gz"])
            with open(os.path.join(output_dir, "splits_final.json")) as f:
                self.assertEqual(json.load(f), [
                    {"train": ["hcms_hc_01_z00", "duke_01_z10"], "val": ["hcms_hc_01_z01"]},
                    {"train": ["hcms_hc_01_z01"], "val": ["hcms_hc_01_z00"]},
                ])

    def _split_semantic_data(self, hcms_case, duke_case):
        image = np.full((8, 16), 7, dtype=np.uint8)
        label = np.zeros((8, 16), dtype=np.uint8)
        label[2:4] = 2
        with tempfile.TemporaryDirectory() as tmp_dir:
            nnunet_dir = os.path.join(tmp_dir, "nnunet")
            umg_dir = os.path.join(tmp_dir, "umg")
            output_dir = os.path.join(tmp_dir, "out")
            os.makedirs(umg_dir)
            _write_nnunet_case(nnunet_dir, hcms_case, image, label)
            _write_nnunet_case(nnunet_dir, duke_case, image, label)
            open(os.path.join(umg_dir, "RP012_2023_OD_z08.h5"), "w").close()
            json_file = os.path.join(tmp_dir, "split.json")
            _write_split(json_file, ["oct_11001_000", "oct_30012_2023_OD_008"], ["oct_20001030", "oct_20002000"])

            missing = split_semantic_data(nnunet_dir, umg_dir, output_dir, json_file)

            self.assertEqual(missing, ["oct_20002000"])
            self.assertEqual(os.listdir(os.path.join(output_dir, "umg_rp_train")), ["RP012_2023_OD_z08.h5"])
            self.assertEqual(os.listdir(os.path.join(output_dir, "duke_dme_val")), [f"{duke_case}.h5"])
            with h5py.File(os.path.join(output_dir, "hcms_train", f"{hcms_case}.h5"), "r") as f:
                self.assertEqual(f["image"].dtype, np.uint8)
                np.testing.assert_array_equal(f["masks"][:], label)
            with open(os.path.join(output_dir, "semantic_training_split.json")) as f:
                split = json.load(f)
            self.assertEqual(split["umg_rp"]["train"], ["oct_30012_2023_OD_008"])

    def test_split_semantic_data_legacy_names(self):
        self._split_semantic_data("oct_11001_000", "oct_20001030")

    def test_split_semantic_data_standard_names(self):
        self._split_semantic_data("hcms_hc_01_z00", "duke_01_z30")

    def test_finetuning_splits_match_doc(self):
        input_json = os.path.join(SPLIT_DIR, "train_splits_iterative_all.json")
        with tempfile.TemporaryDirectory() as tmp_dir:
            create_train_val_splits_for_finetuning(tmp_dir, input_json)
            create_train_val_splits_for_finetuning(tmp_dir, input_json, output_sam=True)

            names = sorted(os.listdir(tmp_dir))
            self.assertEqual(len(names), 12)
            for name in names:
                with open(os.path.join(tmp_dir, name)) as f, open(os.path.join(SPLIT_DIR, name)) as g:
                    self.assertEqual(json.load(f), json.load(g), name)


if __name__ == "__main__":
    unittest.main()
