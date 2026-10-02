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
Use the corrected labels to retrain the model (see [Retraining on annotations](#retraining-on-annotations)).
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

## Retraining on annotations

Use the corrected labels of step 6 to fine-tune the model on your data.
The retrained model is a new nnU-Net model with its own dataset ID.
To retrain a model with several retinal layers, see [Retraining on corrected labels](nnunet.md#retraining-on-corrected-labels).

### Requirements

- An NVIDIA GPU. Training on a CPU is too slow.
- At least 5 corrected B-scans. nnU-Net splits the training data into 5 folds.
- Folders for the nnU-Net training data. Add these lines to your `~/.bashrc` next to `nnUNet_results`, then open a new terminal:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
```

### Retrain the model

```bash
micromamba activate oct-sam
oct_tools.retrain_nnunet -i /path/to/images -l /path/to/corrected -p 013 -d 014
```

- `-i` is the image folder of step 5.
- `-l` is the folder with the corrected labels of step 6.
- `-p` is the dataset ID of the pretrained model.
- `-d` is the dataset ID of the retrained model. Use an ID that does not exist on your system.
  If the ID exists, or a previous run with this ID was interrupted, the command stops and shows the folders of the ID.
  Choose another ID, or delete these folders. A folder in `$nnUNet_results` can contain a trained model.

The command does these steps:
1. It copies the B-scans that have a corrected label into the nnU-Net dataset `$nnUNet_raw/Dataset014_OCT-2d-retrain-013`. B-scans without a corrected label are not used. For a binary model such as `013`, every label value above `0` becomes retina (`1`).
2. It prepares the dataset with the network configuration of the pretrained model.
3. It trains fold 0 and starts with the weights of the pretrained model. Fold 0 uses 80 % of the B-scans for training and 20 % for validation.

The retrained model is in `$nnUNet_results/Dataset014_OCT-2d-retrain-013/nnUNetTrainer__nnUNetPlans__2d`.
The files `fold_0/progress.png` and `fold_0/training_log_*.txt` show the progress of the training.
If the training stops, continue it in the `nnunet` environment:

```bash
micromamba activate nnunet
nnUNetv2_train 014 2d 0 --c
```

### Shorter training

nnU-Net trains for 1000 epochs. This takes several hours on a GPU.
Select a trainer with fewer epochs with `-tr`:

```bash
oct_tools.retrain_nnunet -i /path/to/images -l /path/to/corrected -p 013 -d 014 -tr nnUNetTrainer_250epochs
```

nnU-Net has trainers for 1, 5, 10, 20, 50, 100, 250, 500 and 750 epochs, for example `nnUNetTrainer_100epochs`.
The trainer name is part of the model folder, for example `nnUNetTrainer_250epochs__nnUNetPlans__2d`.
Thus, add the same `-tr` value when you apply the model or continue its training (`nnUNetv2_train 014 2d 0 -tr nnUNetTrainer_250epochs --c`).
A retraining with `-p 014` finds the trainer of model 014 automatically.

### Apply the retrained model

```bash
micromamba activate oct-sam
oct_tools.apply_nnunet -i /path/to/images -o /path/to/segmentations_014 -d 014 --device cuda
```

Add the `-tr` value of the retraining, if you used one.

### Continue the annotation loop

Each loop adds corrected labels and gives a better model:

1. Apply the newest model to all B-scans, as shown above.
2. Correct the new segmentations. Use the same output folder `-o` as before. napari opens only the B-scans that have no corrected label.
   ```bash
   python scripts/mouse/annotate.py -i /path/to/images -l /path/to/segmentations_014 -o /path/to/corrected
   ```
3. Retrain on all corrected labels. Use the newest model for `-p` and a new ID for `-d`.
   ```bash
   oct_tools.retrain_nnunet -i /path/to/images -l /path/to/corrected -p 014 -d 015
   ```
4. Repeat the loop until the segmentations need only few corrections.

You can add new B-scans to the image folder at any time. Step 1 segments them.

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
