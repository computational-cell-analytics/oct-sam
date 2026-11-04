# Training of nnU-Net v2

To install nnU-Net and segment B-scans with the pre-trained model, see [nnunet.md](nnunet.md).
For the datasets, their layers and their voxel sizes, see [training_data.md](training_data.md).

[nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) is the baseline method for the comparison with OCT-SAM.
All networks use the nnU-Net configuration `2d`.

## Networks

The networks were trained in three stages:
1. With the public datasets HCMS and Duke DME only.
2. With the public datasets and the UMG-RP data.
3. From the network of stage 1, with 1, 5, 10, 25, 50 and 100 additional UMG-RP B-scans. All of these networks use the same 10 UMG-RP B-scans for validation. This stage shows how the number of local training samples changes the accuracy.

| Dataset | Training data | Note |
|---|---|---|
| `Dataset001_OCT-2d` | Same network as `Dataset004_OCT-2d-all` | The published model. See [nnunet.md](nnunet.md). |
| `Dataset004_OCT-2d-all` | HCMS, Duke DME and UMG-RP, 5 folds | Stage 2 |
| `Dataset005_OCT-2d-public-pretrain` | HCMS and Duke DME | Stage 1 |
| `Dataset006_OCT-2d-finetune-all` | 005, fine-tuned on all UMG-RP B-scans (179 for training, 45 for validation) | Stage 3 |
| `Dataset007_OCT-2d-finetune-n100` ... `Dataset012_OCT-2d-finetune-n001` | 005, fine-tuned on 100, 50, 25, 10, 5 and 1 UMG-RP B-scans | Stage 3 |
| `Dataset013_OCT-2d-binary` | HCMS, Duke DME and UMG-RP, with one label for the retina | For mouse OCT. See [2d_mouse.md](2d_mouse.md). |

## Accuracy

For a consistent comparison, only fold 0 of each network was evaluated on the validation dataset `20250717`.
- `results/model_accuracy_validation/20250717_nnunet_<ID>-2d_f0.txt` contains the values for each B-scan.
- `results/model_accuracy.xlsx` contains the summary.

The UMG-RP data improves the accuracy very much.
The public datasets always contain all layers.
In the UMG-RP data, the degeneration removes the lower layers (EZ, ONL and OPL) in some B-scans.
A network that has such B-scans in its training data segments these cases much better.

To compare the segmentation with manual thickness measurements, use `scripts/compare_seg_to_thickness_measurement.py`:
```bash
python scripts/compare_seg_to_thickness_measurement.py --measurement results/clinical_validation/thickness_measurement_manual_v2.json --nnunet_dir /path/to/val_seg_004-2d_f0 -o results/clinical_validation/thickness_measurement_nnunet_004-2d_f0.json
```

## Convert the data

nnU-Net needs NIfTI files that contain the voxel size.
The conversion scripts write the folders `imagesTr` and `labelsTr` into the output folder.
Use the output folder as a dataset folder in `$nnUNet_raw`.

Public datasets (see [training_data.md](training_data.md) for the downloads):
```bash
# HCMS
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/HCMS_DATA/OCT_Manual_Delineations-2018_June_29/ -o <OUTPUT_DIR> --dataset hcms
# Duke DME
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/DUKE_DME_DATA/duke_dme_2015_BOE_Chiu2/2015_BOE_Chiu/ -o <OUTPUT_DIR> --dataset duke_dme
```

UMG-RP data in H5 format:
```bash
python scripts/process_nnunet_data/nnunet_preprocess_internal_data.py -i <INPUT_DIR> -o <OUTPUT_DIR> -l edit_v3
```
`-l` selects the label version in the H5 file. See [UMG-RP](training_data.md#umg-rp).

### Binary retinal band

Both conversion scripts accept `--binary`.
This option merges all retinal layers into a single label.
The output label has the value 0 for the background and the value 1 for the retinal layers.
Use it to train a network that segments the retinal band as a whole.
This is necessary for data where the signal-to-noise ratio is too low to separate the individual layers, for example OCT of mice.

```bash
# HCMS
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/HCMS_DATA/OCT_Manual_Delineations-2018_June_29/ -o <OUTPUT_DIR> --dataset hcms --binary
# Duke DME
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/DUKE_DME_DATA/duke_dme_2015_BOE_Chiu2/2015_BOE_Chiu/ -o <OUTPUT_DIR> --dataset duke_dme --binary
# UMG-RP
python scripts/process_nnunet_data/nnunet_preprocess_internal_data.py -i <INPUT_DIR> -o <OUTPUT_DIR> -l edit_v3 --binary
```

For HCMS, `--binary` replaces the combination of the inner and outer photoreceptor segments, because that step has no effect on a binary label.

The scripts do not create a `dataset.json`.
Write it manually with two classes:
```json
{
    "channel_names": {"0": "OCT"},
    "labels": {"background": 0, "retina": 1},
    "numTraining": <NUMBER_OF_FILES_IN_LABELSTR>,
    "file_ending": ".nii.gz"
}
```

## Train with a fixed split

The fine-tuning networks of stage 3 use the splits in `doc/train_splits`. See [Train and validation splits](training_data.md#train-and-validation-splits).

1. Copy the B-scans of a split into a new dataset:
   ```bash
   python scripts/process_nnunet_data/create_retrain_data.py -i "$nnUNet_raw/Dataset006_OCT-2d-finetune-all" -o "$nnUNet_raw/Dataset007_OCT-2d-finetune-n100" -j doc/train_splits/train_splits_n100.json
   ```
2. Plan and preprocess the new dataset:
   ```bash
   nnUNetv2_plan_and_preprocess -d 007 --verify_dataset_integrity
   ```
3. Copy the JSON file to `$nnUNet_preprocessed/Dataset007_OCT-2d-finetune-n100/splits_final.json`.
   nnU-Net then uses this split for fold 0.

To make new splits, use `create_train_val_splits_for_finetuning`.
Write them into a new folder. With the same random seed, the function gives the same splits as the files in `doc/train_splits`.
```python
from oct_tools.train_utils import create_train_val_splits_for_finetuning

create_train_val_splits_for_finetuning("new_splits", "doc/train_splits/train_splits_iterative_all.json")
```

## Retraining on corrected labels

`oct_tools.retrain_nnunet` fine-tunes an installed 2D model on images and corrected labels.
The retrained model is a new nnU-Net model with its own dataset ID.
It has the same labels as the pretrained model.
For the 7-layer models, for example `001`, the labels keep the layer IDs 1-7.

```bash
oct_tools.retrain_nnunet -i <h5_dir> -l <h5_dir> -p 001 -d 014
```

- `-i` and `-l` can be the same folder of H5 files. The command reads the image from `image` and the label from `labels/<label_key>`.
- `--label_key` selects the label version. The default is `edit_v3`.
- TIF labels are also supported, with the same file names as the images.
- The label IDs must be IDs of the pretrained model, as in its `dataset.json`. The command stops at an unknown ID.
- The command does not check which layer an ID marks. Each ID must mark the same layer as in the pretrained model: 1 RNFL, 2 GCIPL, 3 INL, 4 OPL, 5 ONL, 6 EZ, 7 RPE.
- A raw SAM segmentation has arbitrary IDs. Assign the layer IDs first with the post-processing step `assign_layer_id`.

[2d_mouse.md](2d_mouse.md#retraining-on-annotations) describes the requirements, shorter training with `-tr`, the application of the retrained model, and the annotation loop.

## Run nnU-Net directly

`oct_tools.apply_nnunet` converts TIF files to NIfTI, runs `nnUNetv2_predict` in the `nnunet` environment, and converts the result back to TIF.
Use the nnU-Net commands directly only for NIfTI data or for options that `oct_tools.apply_nnunet` does not have.

Predict with fold 0:
```bash
nnUNetv2_predict -d DATASET_NAME_OR_ID -i INPUT_DIR -o OUTPUT_DIR -f 0 -c 2d
```

Predict with all 5 folds of `Dataset004_OCT-2d-all`:
```bash
nnUNetv2_predict -d Dataset004_OCT-2d-all -i INPUT_DIR -o OUTPUT_DIR -f 0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans
```

If all 5 folds of a network are trained, nnU-Net can find the best configuration and its post-processing:
```bash
nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c 2d
```
The command shows the prediction command for this configuration.

## For maintainers

- The data is in `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`.
- The models are in `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`.
- On the cluster, the next free dataset ID is 014.
