# Training data

Three datasets were used to train OCT-SAM and nnU-Net.
This page is the reference for the datasets, their layers, their voxel sizes and the train/validation splits.

| Dataset | Subjects | B-scans used | Voxel size (µm) | Access |
|---|---|---|---|---|
| HCMS | 35: 14 healthy controls, 21 with multiple sclerosis | 1568 (32 volumes x 49 B-scans) | (3.87, 5.8, 123.6) | Public |
| Duke DME | 10 with diabetic macular edema | 110 | (3.87, 11.33) | Public |
| UMG-RP | 37 with retinitis pigmentosa | 224 | (3.87, 5.88) | Private |

The voxel size is in the order (axial, lateral) or (axial, lateral, slice distance).
The data conversion writes these values into the NIfTI files for nnU-Net.

## HCMS: Healthy Control and Multiple Sclerosis

- [Publication](https://doi.org/10.1016/j.dib.2018.12.073), [data](https://medic.rad.jhmi.edu/index.php?title=OCT_Data)
- 35 volumes of Heidelberg Spectralis scans, with 49 B-scans in each volume.
- Each B-scan has 8 annotated layers.
- 3 volumes (MS 14, MS 16, MS 17) are not used, because their control points are missing.
- Resolution (mean and standard deviation):
    - Lateral (between A-scans): 5.8 µm (±0.2)
    - Axial (between two pixels in an A-scan): 3.9 µm (±0.0)
    - Slice distance (between B-scans): 123.6 µm (±3.6)
    - Imaging area: approximately 6 x 6 mm²

## Duke DME

- Publication: S. J. Chiu et al., "Kernel regression based segmentation of optical coherence tomography images with diabetic macular edema", Biomedical Optics Express 6(4), 2015, [doi:10.1364/BOE.6.001172](https://doi.org/10.1364/BOE.6.001172)
- [Data](https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm)
- 10 subjects. 11 B-scans of each volume are annotated, so 110 B-scans in total.
- The annotation contains 7 layers and the fluid regions. The conversion uses only the layers of the first manual annotation (`manualLayers1`).
- Resolution:
    - Lateral: 11.07 to 11.59 µm
    - Axial: 3.87 µm
    - Slice distance: 118 to 128 µm

## UMG-RP

- A retrospective cohort of 37 RP patients of the Department of Ophthalmology, University Medical Center Göttingen, from 2019 to 2025.
- 224 B-scans with a manual annotation of the 7 layers.
- The data is private.

Each B-scan is one H5 file with the name `RP<id>_<year>_<eye>_z<slice>.h5`, for example `RP012_2023_OD_z08.h5`.
`doc/name_mapping.tsv` maps the original acquisition names to these names.

The H5 file contains these datasets:

| Key | Content |
|---|---|
| `image` | The B-scan |
| `labels/original` | The manual annotation |
| `labels/edit_v1` | The annotation after clean-up, gap filling, and restriction to the image area |
| `labels/edit_v2` | The `edit_v1` steps, then the common layer IDs 1-7 |
| `labels/edit_v3` | The same steps as `edit_v2`. This is the label for training. |

`scripts/process_data/convert_annotations.py` writes `image` and `labels/original` from the TIF files of an annotation cycle.
`scripts/process_data/edit_annotations.py -v <version>` writes the `edit_v*` labels.
`scripts/process_data/check_data.py` shows the images and the labels in napari.

The annotation cycles are in these folders:

| Annotation cycle | Folder |
|---|---|
| `annotations_training_cycle_1` | `20251126` |
| `annotations_training_cycle_2` | `20251215` |
| `annotations_training_cycle_3` | `20260105` |

## Mapping to the 7 layers

The models segment 7 layers. The table shows the annotated layers of each dataset for each label ID.

| ID | Layer | HCMS | Duke DME | UMG-RP |
|---|---|---|---|---|
| 1 | RNFL | RNFL | NFL | RNFL |
| 2 | GCIPL | GCL+IPL | GCL-IPL | GCIPL |
| 3 | INL | INL | INL | INL |
| 4 | OPL | OPL | OPL | OPL |
| 5 | ONL | ONL | ONL-ISM (outer nuclear layer and inner segment myeloid) | ONL |
| 6 | EZ | IS + OS (inner and outer photoreceptor segments) | ISE (inner segment ellipsoid) | EZ |
| 7 | RPE | RPE | OS-RPE (outer segment and retinal pigment epithelium) | RPE |

For HCMS, the conversion combines the inner and outer photoreceptor segments into the ellipsoid zone.
With the option `--keep_is_os`, the conversion keeps the 8 HCMS layers. The pre-training of the published OCT-SAM models used the 8 layers.
With the option `--binary`, the conversion merges all layers into the label `1`.

## B-scan names

The data conversion gives each B-scan a standard name with the form `<dataset>_<id>_z<slice>`.
nnU-Net and OCT-SAM use the same names.

| Dataset | Standard name | `<id>` |
|---|---|---|
| HCMS | `hcms_hc_01_z00` | `hc` (healthy control) or `ms` (multiple sclerosis), and the subject number |
| Duke DME | `duke_01_z10` | The subject number |
| UMG-RP | `umgrp_012_2023_OD_z08` | The patient number, the year and the eye |

`<slice>` is the index of the B-scan in the volume, with 2 digits.

With the option `--legacy_names`, the conversion writes the names of the earlier conversions:

| Dataset | Legacy nnU-Net name | Legacy H5 name |
|---|---|---|
| HCMS | `oct_11001_000`, with `11` for `hc` and `12` for `ms` | `hc01_spectralis_macula_v1_s1_R_000` |
| Duke DME | `oct_20001010` | `Subject_01_z000` |
| UMG-RP | `oct_30012_2023_OD_008` | `RP012_2023_OD_z08` |

- The split files `train_splits_nnunet_all.json`, `train_splits_iterative_all.json` and `train_splits_n<N>.json` contain the legacy nnU-Net names.
- The pre-training data of the published OCT-SAM models has the legacy H5 names.
- The legacy H5 name of Duke DME counts only the 11 annotated B-scans of a volume. For example, `Subject_01_z000` is B-scan 10 of the volume, with the standard name `duke_01_z10`.
- The UMG-RP H5 files with the annotations keep the names `RP<id>_<year>_<eye>_z<slice>.h5`.

`standard_name` in `oct_tools/data_processing.py` converts a legacy name to the standard name.
It cannot convert the legacy H5 names of Duke DME.

Do not mix standard names and legacy names in one folder. Otherwise, the folder holds B-scans twice.
The conversion stops if the output folder holds a B-scan with another name.
It also stops if an output H5 file exists and holds other datasets than `image` and `masks`, for example the annotations of a UMG-RP file.

## Data conversion

- For nnU-Net, see [Convert the data](nnunet_training.md#convert-the-data).
- For OCT-SAM, see [Prepare the data](oct-sam_training.md#prepare-the-data).

## Train and validation splits

The folder `doc/train_splits` contains the splits as JSON files.
Each file contains a list of folds. Each fold has a `train` and a `val` list of case names.

| File | Content |
|---|---|
| `train_splits_nnunet_all.json` | 5 folds of all 1902 B-scans of the three datasets: 1521 for training, 381 for validation |
| `train_splits_iterative_all.json` | The 224 UMG-RP B-scans: 179 for training, 45 for validation |
| `train_splits_n<N>.json` | `<N>` UMG-RP B-scans for training and the same 10 B-scans for validation. Legacy nnU-Net names. |
| `train_splits_sam_n<N>.json` | The same splits as `train_splits_n<N>.json`, with the names of the UMG-RP H5 files |

`<N>` is 001, 005, 010, 025, 050 or 100.
`scripts/process_data/split_data.py` finds a B-scan by its standard name or by its legacy name.
Thus each split file works with data that has standard names or legacy names.
`create_train_val_splits_for_finetuning` in `oct_tools/data_processing.py` made the `n<N>` files from `train_splits_iterative_all.json`.
