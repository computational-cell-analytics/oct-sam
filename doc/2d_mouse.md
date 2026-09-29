# 2D segmentation of mouse OCT

Mouse OCT has a much lower signal-to-noise ratio than human OCT.
The individual retinal layers cannot be separated reliably.
Thus, a binary nnU-Net model segments the retinal band as a whole.

## Binary 2D nnU-Net model

| Property | Value |
|---|---|
| Dataset name | `Dataset013_OCT-2d-binary` |
| Configuration | `2d`, fold 0 |
| Labels | `0`: background, `1`: retina |
| Training data | 1902 human B-scans from HCMS, Duke DME and UMG-RP (fold 0: 1521 for training, 381 for validation) |
| Validation Dice | 0.988 (human B-scans) |
| Download | `nnunet_model_013_oct-2d-binary.zip` from [GWDG OwnCloud](https://owncloud.gwdg.de/index.php/s/ogQJT7n8rSRFbhr). |

The training data contains only human OCT.
The labels were merged into a single retinal band with the `--binary` option (see [Binary retinal band](nnunet.md#binary-retinal-band)).
A test on 8 mouse B-scans gave a mean Dice of 0.66 against manual annotations.
The Dice was 0.68–0.93 for the 6 B-scans with low background noise, and 0.16–0.27 for the 2 noisy B-scans.
Check each segmentation visually (see step 6).

## Apply the model to your data

The steps below segment a folder of mouse B-scans in TIF format.
You need two environments:
- `oct-sam` for this repository. It converts the data between TIF and NIfTI.
- `nnunet` for nnU-Net. It runs the network.

### 1. Install the repository

```bash
git clone https://github.com/computational-cell-analytics/oct-analysis.git
cd oct-analysis
conda env create -f environment.yaml
conda activate oct-sam
pip install -e .
```

### 2. Install nnU-Net

```bash
micromamba create -n nnunet python=3.12
micromamba activate nnunet
# select the correct build for your system on https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install nnunetv2
```

nnU-Net reads its models from the directory in the environment variable `nnUNet_results`.
Add this line to your `~/.bashrc`, then open a new terminal:

```bash
export nnUNet_results=/path/to/nnUNet_results
```

The [nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/set_environment_variables.md) describes this for other systems.

### 3. Install the model

Download `nnunet_model_013_oct-2d-binary.zip` from [GWDG OwnCloud](https://owncloud.gwdg.de/index.php/s/ogQJT7n8rSRFbhr).
Then install it:

```bash
micromamba activate nnunet
nnUNetv2_install_pretrained_model_from_zip /path/to/nnunet_model_013_oct-2d-binary.zip
```

The model is now in `$nnUNet_results/Dataset013_OCT-2d-binary`.

### 4. Prepare the data

Put all B-scans into one input folder.
- Use one TIF file for each 2D B-scan, or one TIF file for each stack of B-scans.
- Grayscale and RGB TIF files are both supported. The conversion averages the three RGB channels.
- Put only images into the input folder. The conversion reads every TIF file, including annotation files.

The conversion writes a pixel size of (3.87, 5.88) µm (axial, lateral) into each NIfTI file.
The mouse test data was processed with the same values.
Do not scale the B-scans before the segmentation.

### 5. Segment the data

```bash
micromamba activate oct-sam
oct_tools.apply_nnunet -i /path/to/images -o /path/to/segmentations -d 013
```

- Add `--device cuda` if you have an NVIDIA GPU. The default is `cpu`.
- Add `-m conda` if you installed the `nnunet` environment with conda.

The command writes one segmentation `<name>.tif` for each image `<name>.tif`.
A stack `<name>.tif` gives one segmentation `<name>_z<slice>.tif` for each B-scan.
Each segmentation has the values `0` (background) and `1` (retina).

### 6. Correct the segmentation

Correct the nnU-Net segmentations in napari.
The corrected labels can become training data for a mouse-specific model.
Run the script from the repository folder:

```bash
micromamba activate oct-sam
python scripts/mouse/annotate.py -i /path/to/images -l /path/to/segmentations -o /path/to/corrected
```

- `-i` is the image folder of step 5. The script reads 2D B-scans and stacks.
- `-l` is the segmentation folder of step 5. The script stops if a segmentation is missing.
- `-o` is the output folder for the corrected labels.

napari opens the first B-scan that has no corrected label.
The layer `committed_objects` contains the nnU-Net segmentation.
Correct it in one of these ways:
- Paint or erase in `committed_objects` with the napari label tools.
- Segment the retina with SAM. Draw a box or set points, then commit the object. The script uses the SAM model `vit_b_medical_imaging`.

Press `N` (**Next Image**) to save the label and open the next B-scan.
The label of `<name>.tif` is saved as `<name>.tif` in the output folder.
Each B-scan of a stack is saved separately as `<name>_z<slice>.tif`.

You can close napari at any time.
The next start continues with the first B-scan that has no corrected label.
To correct a B-scan again, delete its label in the output folder.

A committed SAM object gets a new label ID.
Every label value above `0` counts as retina.

The first start downloads the SAM model.
SAM computes the embedding of each B-scan when the B-scan opens. This takes a few seconds on a CPU.

### 7. Check the result

Export an overlay of the image and the segmentation:

```bash
oct_tools.export_annotations -i /path/to/images/<name>.tif -s /path/to/segmentations/<name>.tif -o <name>_overlay.png --mode rgb
```

Without `--mode rgb`, the command writes an ImageJ composite TIF, in which you can show or hide the segmentation.

## Export the model

This section is for maintainers.
The nnU-Net export contains files with cluster paths, the host name, and the names of the validation cases.
nnU-Net does not need these files for inference or fine-tuning.
Remove them from the zip before you publish it:

| File | Content | Action |
|---|---|---|
| `fold_0/debug.json` | Cluster paths, host name, GPU | Remove |
| `fold_0/validation/summary.json` | Cluster paths, names of the validation cases | Remove |
| `fold_0/checkpoint_final.pth` | Network weights and training state, no paths | Keep |
| `plans.json`, `dataset.json` | Network configuration and labels, required for inference | Keep |
| `dataset_fingerprint.json`, `fold_0/progress.png` | Image statistics, training curves | Keep |

```bash
nnUNetv2_export_model_to_zip -d 013 -c 2d -f 0 -o model_013_2d-binary_f0.zip
cp model_013_2d-binary_f0.zip nnunet_model_013_oct-2d-binary.zip
zip -d nnunet_model_013_oct-2d-binary.zip '*/debug.json' '*/validation/summary.json'
# check: no match
unzip -p nnunet_model_013_oct-2d-binary.zip | grep -caE '/mnt|/home|oct_30'
```
