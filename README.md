# Analysis of Retinal Layers with OCT-SAM and nnU-Net

Segmentation and measurements for retinal layers in OCT data using neural networks, e.g. OCT-SAM and nnU-Net.

# Installation

## OCT-SAM
OCT-SAM can be installed via `conda` (or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)). To install it:

Download the github repository:
```bash
git clone https://github.com/computational-cell-analytics/oct-sam
```
Go to the directory:
```bash
cd oct-sam
```
Create an environment with the required dependencies:
```bash
conda env create -f environment.yaml
```
Activate the environment:
```bash
conda activate oct-sam
```
Install the oct_tools package:
```bash
pip install -e .
```
(Optional) To process external data from the public dataset, which is used for training, `eyepy` is required ([Github](https://github.com/MedVisBonn/eyepy)). It can be installed with:
```bash
pip install -U eyepy
```

## nnU-Net

Follow the instructions specified in `doc/nnunet.md`.

# Usage

The relevant functions are:
- `oct_tools.interactive`: For automatic and interactive segmentation.
    - Automatic segmentation is based on deriving prompts from the SAM predictions and then segmenting the layers with the fine-tuned SAM model based on these prompts.
- `oct_tools.metrics`: Calculate metrics for a segmentation.
- `oct_tools.measure`: Interactive measurement tool for segmentation analysis with napari.
- `oct_tools.apply_sam`: Apply an OCT-SAM model on multiple images without interactions.
- `oct_tools.apply_nnunet`: Apply the pre-trained nnU-Net model on multiple images without interactions.
- `oct_tools.eval_segmentation`: Evaluate segmentation by comparing it to labels to measure network performance.
- `oct_tools.open_labels`: Open one or more segmentations in napari with the retinal layer color map.

The following scripts are relevant:
- `scripts/training/pretrain_oct_sam_on_public_datasets.py`: For pre-training a SAM model on the public datasets.
- `scripts/training/train_oct_sam.py`: For fine-tuning a SAM model on the public datasets and the UMG-RP data.
- `scripts/training/finetune_pretrained_model_iteratively.py`: For fine-tuning a pre-trained checkpoint on a data subset.
- `scripts/training/train_oct_sam_semantic.py`: For training a SAM model for semantic segmentation.

The data is located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`. Currently (2026-04-29), it is not clear if the data will be published.

The models are located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`.
Models available for download online:
- `oct-sam-V1.pt`: The fine-tuned SAM model, which was trained on public datasets (HCMS and Duke DME) and the UMG-RP data. [Download from ownCloud](https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA)

# Data

Two public datasets, HCMS and Duke DME, and a private dataset UMG-RP were used for network training.
UMG-RP consists of a retrospective cohort of 37 retinitis pigmentosa (RP) patients who presented at the Department of Ophthalmology, University Medical Center of Göttingen, between 2019 and 2025.
Detailed information about the data can be found here: `doc/training_data.md`

## Retinal Layers
The segmentation data was limited to the 7 layers:
* RNFL: Retina nerve fiber layer
* GCL+IPL: Ganglion cell layer and inner plexiform layer
* INL: Inner nuclear layer
* OPL: Outer plexiform layer
* ONL: Outer nuclear layer
* EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
* RPE: Retinal pigment epithelium

# Segmentation models

Two segmentation models were used:
- OCT-SAM:
    - model for interactive segmentation
	- based on [µSAM](https://doi.org/10.1038/s41592-024-02580-4) (Segment Anything Model for microscopy)
- nnU-Net:
    - open source network for 2D and 3D semantic segmentation in medical imaging
    - [paper](https://doi.org/10.1038/s41592-020-01008-z)

The pre-trained models can be downloaded here:
- [OCT-SAM-V1](https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA)
- [nnU-Net](https://owncloud.gwdg.de/index.php/s/ffZ4MvFWt8E5jpv)

Additional information about network training and application can be found here: `doc/oct-sam.md` and `doc/nnunet.md`.
