# Training and evaluation of octSAM

MedicoSAM was chosen as a base for the further training of octSAM. [MedicoSAM](https://pubmed.ncbi.nlm.nih.gov/41406266/)

It was re-trained for these conditions:
* First, only based on two public datasets (`oct-sam-pretrained-v2`):
    * Duke_DME(110 B-scans, 7–8 layers + fluids), [publication](https://doi.org/10.1117/12.2654210), [data](https://people.duke.edu/~sf59/software.html)
    * HCMS (35 vols, 8 layers each B-scan), [publication](https://www.sciencedirect.com/science/article/pii/S2352340918316135), [data](https://medic.rad.jhmi.edu/index.php?title=OCT_Data)
* Secondly, with the addition of custom training data of the project (7 layers, 224 B-scans), `oct-sam-v7`
* Lastly, based on the pre-trained network (`oct-sam-pretrained-v2`), several networks were trained with an iteratively increasing number of samples to test the influence of an increase in the number of local datasets. The networks were trained with 1, 5, 10, 25, 50, and 100 additional images, which are subsets from the custom training data. The same set of 10 validation images was used for validation during training. The train splits can be found under `doc/train_splits`.

## Training data

### Layers
The segmentation of the custom project data was limited to the 7 layers:
* RNFL: Retina nerve fiber layer
* GCL+IPL: Ganglion cell layer and inner plexiform layer
* INL: Inner nuclear layer
* OPL: Outer plexiform layer
* ONL: Outer nuclear layer
* EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
* RPE: Retinal pigment epithelium

For HCMS, the inner and outer photoreceptor segments were combined to the ellipsoid zone.

### Voxel size
* HCMS: (3.87, 5.8, 123.6) µm
* DUKE_DME: (3.87, 11.33) µm
* custom data: (3.87, 5.88) µm

### Public datasets
The public datasets can be downloaded with the links referenced above.

### Internal data
The internal data is stored in h5 files.
They contain different versions of the labels, e.g. the original annotations are stored under `["labels"]["orig"]`.
The annotations can be further refined by removing stray pixels, restricting the labels to the image, and synchronizing the label IDs, so that the same label ID always corresponds to the same semantic layer.
```bash
python scripts/process_nnunet_data/nnunet_preprocess_internal_data.py -i <INPUT_DIR> -o <OUTPUT_DIR> -l edit_v3
```

## Training scripts

`oct-sam-pretrained-v2`:
```bash
python /path/to/oct-analysis/scripts/training/pretrain_medicosam.py
```

`oct-sam-v7`:
```bash
python /path/to/oct-analysis/scripts/training/train_medicosam.py
```

`oct-sam-pre-v2-n001`:
```bash
IPUT_TRAIN="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/split_data/train_n001"
IPUT_VAL="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/split_data/val"
MODEL_NAME="oct-sam-pre-v2-n001"
CHECKPOINT=/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models/oct-sam-pretrained-v2.pt

python /path/to/oct-analysis/scripts/training/retrain_medicosam.py -t "$IPUT_TRAIN" -v "$IPUT_VAL" -m "$MODEL_NAME" -c "$CHECKPOINT"
```

## Running inference

The trained octSAM networks are applied using `/path/to/oct-analysis/scripts/sam/eval_sam.py`.
Different settings can be chosen for the application.
The default pipeline involves an initial prediction of the network, which is used to derive point prompts for a second application.
This can be switched off using the `--no_prompts` argument.
Additionally, the prediction can be post-processed using post-processing functions.
After testing several configurations on the validation dataset, it seems that the optimal post-processing is achieved by merging the horizontal layer predictions and subsequently filtering out layers thinner than 5 pixels.

```bash
OCT_DIR="/path/to/oct-analysis"
DATA_DIR="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/validation_data/standard_20250717"
MODEL="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models/oct-sam-v7.pt"
INFERENCE_DIR="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/oct-sam_inference/oct-sam-v7"

# using no point prompts from initial prediction
python "$OCT_DIR"/scripts/sam/eval_sam.py --input "$DATA_DIR" \
    --model "$MODEL" \
    -o "$ODIR" \
    --no_prompts \
    --label_key edit_v3

# using point prompts but no post-processing
python "$OCT_DIR"/scripts/sam/eval_sam.py --input "$DATA_DIR" \
    --model "$MODEL" \
    -o "$ODIR" \
    --label_key edit_v3

# using point prompts and post-processing
python "$OCT_DIR"/scripts/sam/eval_sam.py --input "$DATA_DIR" \
    --model "$MODEL" \
    -o "$ODIR" \
    --postprocess --postprocess_functions merge_horizontal filter_thin \
    --label_key edit_v3
```

## Evaluate iterative prompting of octSAM
The SAM derivatives (µSAM, MedicoSAM, octSAM) are designed for an iterative approach to improve the segmentation.
The user can refine the initial segmentation by providing bounding poxes or point prompts using positive and negative markers.
To emulate this process, the following function was used:

```bash
python /path/to/oct-analysis/scripts/sam/eval_iterative_prompting.py -m vit_b -i /path/to/oct-data/20250717_images -l /path/to/oct-data/20250717_labels --checkpoint /path/to/oct-models/oct-sam-pre-v2-n100.pt -o /path/to/oct-data/eval_interactive/oct-sam-pre-v2-n100
```
