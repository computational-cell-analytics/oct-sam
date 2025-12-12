# OCT Analysis Toolkit

Segmentation and measurements for retinal layers in OCT data.

## Literature:

ChatGPT overview: https://chatgpt.com/share/6926b1ee-8ce8-8000-b4b9-1adc54159d74

Recommended data for pre-training:
- OCT5k (multi-disease, 5 layers, strong multi-grader).
    - https://www.nature.com/articles/s41597-024-04259-z
    - https://doi.org/10.5522/04/22128671
    - I checked this out and converted it, but it's not detailed enough in the layer annotations.
      Pretraining on this data would be detrimental.
- HCMS (35 vols, 9 layers each B-scan).
    - https://www.sciencedirect.com/science/article/pii/S2352340918316135
    - https://medic.rad.jhmi.edu/index.php?title=OCT_Data
- Duke DME (110 B-scans, 7–8 layers + fluids).
    - https://biblio.imi.uni-luebeck.de/pdf/SPIE2023_CAD_Kepp.pdf
    - https://people.duke.edu/~sf59/software.html
- OCTID normals (25 images with layers).
    - https://www.sciencedirect.com/science/article/pii/S0045790618330842

All the preprocessed pre-training data is located at:
`/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data`

## Overview

The following scripts are relevant:
- `scripts/training/finetune_medicosam.py`: For fine-tuning a SAM model for interactive segmentation.
- `scripts/training/training_distances.py`: For training a U-Net for foreground and distance prediction.
- `run_segmentation_interactive.py`: For automatic and interactive segmentation.
    - Automatic segmentation is based on deriving prompts from the SAM predictions and then segmenting the layers with the fine-tuned SAM model based on these prompts.
- `scripts/calculate_metrics.py`: For calculating metrics for a segmentation.

The data is located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`.

The models are located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`. Currently there are the models:
- `oct-2d-v2.pt`: the U-Net for distance prediction, used for deriving prompts (V2).
- `oct-sam-v3.pt`: The fine-tuned SAM model (V3).


## Meeting Notes

Next steps Oct 8:
- Investigate distance target and optimize for a better seeding
- Implement measurements:
    - Layer width: closest distance from upper to lower border. Can implement via distance trafo.
        - For now: statistics over this
        - In the future: measure at given radii for some of the layers
    - Layer length: parametrized by the main center line.
    - Layer area
