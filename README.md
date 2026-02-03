# OCT Analysis Toolkit

Segmentation and measurements for retinal layers in OCT data.

## Literature:

ChatGPT overview: https://chatgpt.com/share/6926b1ee-8ce8-8000-b4b9-1adc54159d74


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

## Data

### Pre-Training

Recommended data for pre-training:
- OCT5k (multi-disease, 5 layers, strong multi-grader).
    - https://www.nature.com/articles/s41597-024-04259-z
    - https://doi.org/10.5522/04/22128671
    - I checked this out and converted it, but it's not detailed enough in the layer annotations.
      Pretraining on this data would be detrimental.
- HCMS (35 vols, 8 layers each B-scan).
    - https://www.sciencedirect.com/science/article/pii/S2352340918316135
    - https://medic.rad.jhmi.edu/index.php?title=OCT_Data
    - Featured layers:
        - RNFL: Retina nerve fiber layer
        - GCL+IPL: Ganglion cell layer and inner plexiform layer
        - INL: Inner nuclear layer
        - OPL: Outer plexiform layer
        - ONL: Outer nuclear layer
        - IS: Inner photoreceptor segments
        - OS: Outer photoreceptor segments
        - RPE: Retinal pigment epithelium
    - Resolution (mean and standard deviation):
        - Lateral resolution (between A-scans) is 5.8 µm (±0.2)
        - Axial resolution (between two pixels in an A-scan) is 3.9 µm (±0.0)
        - Through-plane distance (slice separation) is 123.6 µm (±3.6) between images
        - Imaging area of approximately 6 x 6 mm²

- Duke DME (110 B-scans, 7–8 layers + fluids).
    - https://biblio.imi.uni-luebeck.de/pdf/SPIE2023_CAD_Kepp.pdf
    - https://people.duke.edu/~sf59/software.html
    - Featured layers:
        - NFL: Nerve fiber layer
        - GCL-IPL: Ganglion cell layer - Inner plexiform layer
        - INL: Inner nuclear layer
        - OPL: Outer plexiform layer
        - ONL-ISM: Outer nuclear layer - Inner segment myeloid
        - ISE: Inner segment ellipsoid
        - OS-RPE: Outer segment - Retinal pigment epithelium
    - Resolution:
        - Lateral resolution: ranging from 11.07 - 11.59 µm/pixel
        - Axial resolution: 3.87 µm/pixel
        - Azimuthal resolution: ranging from 118 - 128 µm/pixel

- OCTID normals (25 images with layers).
    - https://www.sciencedirect.com/science/article/pii/S0045790618330842

All the preprocessed pre-training data is located at:
`/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data`

### Training data

Different batches with images and annotations.
Annotations are refined to remove solitary pixels, fill holes, restrict labels to the image, and unify labeling to same classes.
The layers are:
    - RNFL: Retina nerve fiber layer
    - GCL+IPL: Ganglion cell layer and inner plexiform layer
    - INL: Inner nuclear layer
    - OPL: Outer plexiform layer
    - ONL: Outer nuclear layer
    - EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
    - RPE: Retinal pigment epithelium

## Meeting Notes

Next steps Oct 8:
- Investigate distance target and optimize for a better seeding
- Implement measurements:
    - Layer width: closest distance from upper to lower border. Can implement via distance trafo.
        - For now: statistics over this
        - In the future: measure at given radii for some of the layers
    - Layer length: parametrized by the main center line.
    - Layer area
