# Analysis of Retinal Layers with OCT-SAM and nnU-Net

This repository segments and measures the retinal layers in optical coherence tomography (OCT) B-scans.
It was made for patients with retinitis pigmentosa (RP).
Two models segment 7 retinal layers: OCT-SAM and nnU-Net.
The Python package `oct_tools` contains the commands for segmentation, correction in napari, measurement and evaluation.

# Installation

The commands use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
If you use conda, replace `micromamba` with `conda` in the commands below.
The commands download the repository, create the environment `oct-sam` and install the `oct_tools` package.
A repeated run skips the completed steps:
```bash
[ -d oct-sam ] || git clone https://github.com/computational-cell-analytics/oct-sam.git
cd oct-sam
if ! micromamba run -n oct-sam pip show oct-sam >/dev/null 2>&1; then
    micromamba env create -y -f environment.yaml
    micromamba run -n oct-sam pip install -e .
fi
micromamba activate oct-sam
```

nnU-Net needs a second environment. Steps 3 to 5 of [doc/nnunet.md](doc/nnunet.md) install it.

# Get started

Step-by-step guides:
- [doc/nnunet.md](doc/nnunet.md): Install nnU-Net, segment a B-scan with the pre-trained model, and show the result in napari.
- [doc/oct-sam.md](doc/oct-sam.md): Segment a B-scan with OCT-SAM automatically and correct it interactively in napari.
- [doc/measurement.md](doc/measurement.md): Measure the retinal layers of a segmentation.
- [doc/2d_mouse.md](doc/2d_mouse.md): Segment the retina of mouse OCT, correct the segmentation and retrain the model.

Background:
- [doc/training_data.md](doc/training_data.md): The training datasets.
- [doc/nnunet_training.md](doc/nnunet_training.md) and [doc/oct-sam_training.md](doc/oct-sam_training.md): The training of the models.
- [doc/development.md](doc/development.md): Tests, package layout, data flow and file conventions.

# Input data

- Use TIF files. A file contains one 2D B-scan `(y, x)` or a stack of B-scans `(z, y, x)`.
- Use grayscale B-scans. `oct_tools.apply_nnunet` also accepts RGB B-scans and averages the three channels.
- Do not scale the B-scans.

All tools use the pixel size of the Heidelberg Spectralis: 3.87 µm axial and 5.88 µm lateral.
- The nnU-Net conversion writes this pixel size into the NIfTI files.
- The napari measurement (`oct_tools.measure`, `oct_tools.interactive`) always uses this pixel size. You cannot change it.
- `oct_tools.metrics -v` and `oct_tools.measure_binary -a` accept a different pixel size.

For a device with a different pixel size, the thicknesses and areas of the napari measurement are not correct.

Convert a Heidelberg Heyex tomogram in VOL format into a TIF stack:
```bash
oct_tools.convert_vol -i /path/to/tomogram.vol -o data/tomogram.tif
```
For a folder of VOL files, give an input folder and an output folder: `oct_tools.convert_vol -i /path/to/vol_folder -o data/tif_folder`.
The command prints the pixel size of each tomogram. If it is not the Spectralis pixel size above, give it to `oct_tools.metrics -v`.

`oct_tools.apply_nnunet` segments each B-scan of the stack.
`oct_tools.apply_sam` also segments each B-scan of the stack. Its `-z` selects B-scans.
The napari commands select one B-scan: `-z` for `oct_tools.interactive` and `oct_tools.measure`, `--slice` for `oct_tools.open_labels`.

# Commands

The installation adds these commands. Each command shows its options with `--help`.

| Command | Purpose | Guide |
|---|---|---|
| `oct_tools.apply_nnunet` | Segment a TIF, H5 or VOL file, or a folder of these files, with nnU-Net. | [nnunet.md](doc/nnunet.md) |
| `oct_tools.retrain_nnunet` | Fine-tune an nnU-Net model on corrected labels. | [2d_mouse.md](doc/2d_mouse.md#retraining-on-annotations) |
| `oct_tools.interactive` | Segment a B-scan with OCT-SAM, correct it and measure it in napari. | [oct-sam.md](doc/oct-sam.md) |
| `oct_tools.apply_sam` | Segment TIF, H5 or VOL files with OCT-SAM without napari. | [oct-sam.md](doc/oct-sam.md#batch-segmentation) |
| `oct_tools.open_labels` | Show segmentations in napari with the layer colors. | [nnunet.md](doc/nnunet.md#7-show-the-result-in-napari) |
| `oct_tools.measure` | Measure a B-scan and its segmentation in napari. | [measurement.md](doc/measurement.md) |
| `oct_tools.metrics` | Measure a segmentation without napari. Write a TSV or XLSX file. | [measurement.md](doc/measurement.md#calculate-metrics-without-napari) |
| `oct_tools.measure_binary` | Measure the retina thickness of binary segmentations. Write an XLSX file. | [2d_mouse.md](doc/2d_mouse.md#7-measure-the-retinal-thickness) |
| `oct_tools.export_annotations` | Export a B-scan and its segmentation as an ImageJ TIF or an RGB overlay. | [measurement.md](doc/measurement.md#share-annotations-without-napari) |
| `oct_tools.convert_vol` | Convert Heidelberg Heyex VOL files into TIF stacks. | [Input data](#input-data) |
| `oct_tools.eval_segmentation` | Compare segmentations with labels: precision, recall, F1 score, Dice. | [measurement.md](doc/measurement.md#compare-segmentations-with-labels) |

# Models

| Model | Output | Download | Guide |
|---|---|---|---|
| OCT-SAM `oct-sam-V1` | 7 layers, interactive correction | [ownCloud](https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA) | [oct-sam.md](doc/oct-sam.md) |
| nnU-Net `Dataset001_OCT-2d` | 7 layers | [ownCloud](https://owncloud.gwdg.de/index.php/s/ffZ4MvFWt8E5jpv) | [nnunet.md](doc/nnunet.md) |
| nnU-Net `Dataset013_OCT-2d-binary` | Retina as one label, for mouse OCT | [ownCloud](https://owncloud.gwdg.de/index.php/s/ogQJT7n8rSRFbhr) | [2d_mouse.md](doc/2d_mouse.md) |

- OCT-SAM is a fine-tuned [MedicoSAM](https://doi.org/10.1109/TMI.2025.3644811) model. MedicoSAM is based on [µSAM](https://doi.org/10.1038/s41592-024-02580-4) (Segment Anything for Microscopy).
- [nnU-Net](https://doi.org/10.1038/s41592-020-01008-z) is an open-source network for semantic segmentation in medical imaging. It is the baseline.

On the validation data, nnU-Net was more accurate than OCT-SAM:

| Model | F1 score | Symmetric best Dice |
|---|---|---|
| nnU-Net | 0.970 | 0.880 |
| OCT-SAM | 0.851 | 0.766 |

`results/model_accuracy.xlsx` contains all values.

# Retinal layers

The segmentation contains 7 layers. Label `0` is the background.

| ID | Layer | Name |
|---|---|---|
| 1 | RNFL | Retinal nerve fiber layer |
| 2 | GCIPL | Ganglion cell layer and inner plexiform layer |
| 3 | INL | Inner nuclear layer |
| 4 | OPL | Outer plexiform layer |
| 5 | ONL | Outer nuclear layer |
| 6 | EZ | Ellipsoid zone (inner and outer photoreceptor segments) |
| 7 | RPE | Retinal pigment epithelium |

# Data

Two public datasets, HCMS and Duke DME, and the private dataset UMG-RP were used for training.
UMG-RP contains a retrospective cohort of 37 RP patients of the Department of Ophthalmology, University Medical Center Göttingen, from 2019 to 2025.
On 2026-04-29, it was not clear if UMG-RP will be published.
[doc/training_data.md](doc/training_data.md) describes the datasets.
