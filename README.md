# Analysis of Retinal Layers with OCT-SAM

Segmentation and measurements for retinal layers in OCT data using neural networks, e.g. OCT-SAM and nnU-Net.

## Installation

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
conda activate cochlea-net
```
Install the oct_tools package:
```bash
pip install -e .
```
(Optional) To process external data from the public dataset, `eyepy` is required ([Github](https://github.com/MedVisBonn/eyepy)). It can be installed with:
```bash
pip install -U eyepy
```

## Usage

The relevant functions are:
- `oct_tools.interactive`: For automatic and interactive segmentation.
    - Automatic segmentation is based on deriving prompts from the SAM predictions and then segmenting the layers with the fine-tuned SAM model based on these prompts.
- `oct_tools.metrics`: Calculate metrics for a segmentation.
- `oct_tools.measure`: Interactive measurement tool for segmentation analysis with napari.
- `oct_tools.apply_sam`: Apply an OCT-SAM model on multiple images without interactions.
- `oct_tools.eval_segmentation`: Evaluate segmentation by comparing it to labels to measure network performance.
The following scripts are relevant:
- `scripts/training/finetune_medicosam.py`: For fine-tuning a SAM model for interactive segmentation.
- `scripts/training/training_distances.py`: For training a U-Net for foreground and distance prediction.

The data is located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`. Currently (2026-04-29), it is not clear if the data will be published.

The models are located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`.
Models available for download online:
- `oct-sam-V1.pt`: The fine-tuned SAM model, which was trained on public datasets (HCMS and Duke DME) and the UMG-RP data. [Download from ownCloud](https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA)

## Data

Two public datasets, HCMS and Duke DME, and a private dataset UMG-RP were used for network training.
UMG-RP consists of a retrospective cohort of 37 retinitis pigmentosa (RP) patients who presented at the Department of Ophthalmology, University Medical Center of Göttingen, between 2019 and 2025.
Detailed information about the data can be found here: `docs/training_data.md`

### Retinal Layers
The segmentation data was limited to the 7 layers:
* RNFL: Retina nerve fiber layer
* GCL+IPL: Ganglion cell layer and inner plexiform layer
* INL: Inner nuclear layer
* OPL: Outer plexiform layer
* ONL: Outer nuclear layer
* EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
* RPE: Retinal pigment epithelium

## Segmentation models

Two segmentation models were used:
- OCT-SAM:
    - model for interactive segmentation
	- based on [µSAM](https://doi.org/10.1038/s41592-024-02580-4) (Segment Anything Model for microscopy)
- nnU-Net:
    - open source network for 2D and 3D semantic segmentation in medical imaging
    - [paper](https://doi.org/10.1038/s41592-020-01008-z)

Additional information about network training and application can be found here: `docs/oct-sam.md` and `docs/nnunet.md`.
