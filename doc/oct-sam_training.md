# Training and evaluation of OCT-SAM

To install OCT-SAM and segment B-scans with the trained model, see [oct-sam.md](oct-sam.md).
For the datasets, their layers and their voxel sizes, see [training_data.md](training_data.md).

OCT-SAM is a fine-tuned [MedicoSAM](https://doi.org/10.1109/TMI.2025.3644811) model with the model type `vit_b`.
MedicoSAM is based on [µSAM](https://doi.org/10.1038/s41592-024-02580-4) (Segment Anything for Microscopy).

## Models

The models were trained in three stages:

| Model | Training data | Script |
|---|---|---|
| `oct-sam-pretrained-V1` | HCMS and Duke DME | `scripts/training/pretrain_oct_sam_on_public_datasets.py` |
| `oct-sam-V1` | HCMS, Duke DME and UMG-RP. This is the published model. | `scripts/training/train_oct_sam.py` |
| `oct-sam-pre-v2-n001` ... `-n100` | `oct-sam-pretrained-V1`, fine-tuned on 1, 5, 10, 25, 50 and 100 UMG-RP B-scans. All use the same 10 UMG-RP B-scans for validation. | `scripts/training/finetune_pretrained_model_iteratively.py` |

`scripts/training/train_oct_sam_semantic.py` trains a variant for semantic segmentation. This variant is not published.

## Prepare the data

OCT-SAM reads H5 files with the keys `image` and a label key.

Public datasets (see [training_data.md](training_data.md) for the downloads):
```bash
python scripts/process_train_data/preprocess_external_data.py -i /path/to/HCMS_DATA/OCT_Manual_Delineations-2018_June_29/ -o <OUTPUT_DIR>/hcms --dataset hcms
python scripts/process_train_data/preprocess_external_data.py -i /path/to/DUKE_DME_DATA/duke_dme_2015_BOE_Chiu2/2015_BOE_Chiu/ -o <OUTPUT_DIR>/duke_dme --dataset duke_dme
```
The output H5 files contain the label in the key `masks`.

The UMG-RP H5 files need no conversion. The training reads `labels/edit_v3`. See [UMG-RP](training_data.md#umg-rp).

For the fine-tuning of stage 3, copy the B-scans of a split into a folder `train_n<N>` and a folder `val`:
```bash
python scripts/process_nnunet_data/create_retrain_data.py --sam -i <UMG_RP_H5_DIR> -o split_data -j doc/train_splits/train_splits_sam_n001.json
```
`create_train_val_splits_for_finetuning(..., output_sam=True)` in `oct_tools/train_utils.py` makes new split files with the UMG-RP H5 names.

For the semantic variant, `scripts/process_train_data/create_semantic_training_data.py` makes the training data from the nnU-Net dataset and a split file.

## Train

The scripts for stage 1 and stage 2 read the data from fixed cluster paths. Before you run them on a different computer, change the constants `ROOT_DME`, `ROOT_HCMS` and `ROOT_UMG_RP` at the top of the script.

```bash
# Stage 1: oct-sam-pretrained-V1
python scripts/training/pretrain_oct_sam_on_public_datasets.py -m oct-sam-pretrained-V1

# Stage 2: oct-sam-V1
python scripts/training/train_oct_sam.py -m oct-sam-V1

# Stage 3: oct-sam-pre-v2-n001
python scripts/training/finetune_pretrained_model_iteratively.py -t split_data/train_n001 -v split_data/val -m oct-sam-pre-v2-n001 -c oct-sam-pretrained-V1.pt
```

Each script writes the checkpoints to `checkpoints/<model>/` and exports the model to `<model>.pt` in the current folder.

## Accuracy

- `results/model_accuracy_validation/20250717_sam_*.txt` contains the values for each B-scan of the validation dataset `20250717`.
- `results/model_accuracy.xlsx` contains the summary.

The names of these files use the internal model names. See [For maintainers](#for-maintainers).

## Evaluate iterative prompting

With a SAM model, the user can correct a segmentation step by step with point prompts or box prompts.
`scripts/oct-sam/eval_iterative_prompting.py` simulates these corrections and measures the accuracy after each step:
```bash
python scripts/oct-sam/eval_iterative_prompting.py -d <H5_DIR> -i <IMAGE_DIR> -l <LABEL_DIR> -c oct-sam-V1.pt -o <OUTPUT_DIR>
```

- `-d` is optional. With `-d`, the script first writes the images and the `edit_v3` labels of the H5 files as TIF files into `-i` and `-l`.
- `--box` starts with a box prompt. The default is a point prompt.

`results/iterative_prompts/` contains the results.

## For maintainers

The data is in `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`.
The models are in `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`.

Stage 3 on the cluster:
```bash
INPUT_TRAIN="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/split_data/train_n001"
INPUT_VAL="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/split_data/val"
MODEL_NAME="oct-sam-pre-v2-n001"
CHECKPOINT=/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models/oct-sam-pretrained-v2.pt

python scripts/training/finetune_pretrained_model_iteratively.py -t "$INPUT_TRAIN" -v "$INPUT_VAL" -m "$MODEL_NAME" -c "$CHECKPOINT"
```

The files in `results/` use internal model names:

| Internal name | Published name |
|---|---|
| `sam_v7` | `oct-sam-V1` |
| `sam_pretrained-v2`, `oct-sam-pretrained-v2` | `oct-sam-pretrained-V1` |
| `sam_pre-v2-n<N>` | `oct-sam-pre-v2-n<N>` |
| `v6c` | An older model. It is not published. |

The suffixes in `results/model_accuracy_validation/`:
- `prompts`: two-phase inference with prompts.
- `no-prompts`: one-phase inference (`--no_prompts`).
- `pp`: with post-processing.
