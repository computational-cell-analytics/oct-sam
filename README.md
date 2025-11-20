# OCT Analysis Toolkit

Segmentation and measurements for retinal layers in OCT data.

## Overview

The following scripts are relevant:
- training/finetune_medicosam.py: For fine-tuning a SAM model for interactive segmentation.
- training/training_distances.py: For training a U-Net for foreground and distance prediction.
- sam/apply_sam.py: For automatic and interactive segmentation.
    - Automatic segmentation is based on deriving prompts from the U-Net predictions and then segmenting the layers with the fine-tuned SAM model based on these prompts.

The data is located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data`.

The models are located at `/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models`. Currently there are the models:
- `oct-2d-v2.pt`: the U-Net for distance prediction, used for deriving prompts (V2).
- `oct-sam-v1.pt`: The fine-tuned SAM model (V1).


## Meeting Notes

Next steps Oct 8:
- Investigate distance target and optimize for a better seeding
- Implement measurements:
    - Layer width: closest distance from upper to lower border. Can implement via distance trafo.
        - For now: statistics over this
        - In the future: measure at given radii for some of the layers
    - Layer length: parametrized by the main center line. 
    - Layer area
