# Training of nnU-Net v2

nnU-Net v2 was chosen as the baseline method for comparison ([GitHub repository](https://github.com/MIC-DKFZ/nnUNet/tree/master)).

It was trained for two different conditions:
* First, only based on two public datasets:
    * Duke_DME(110 B-scans, 7–8 layers + fluids), [publication](https://doi.org/10.1117/12.2654210), [data](https://people.duke.edu/~sf59/software.html)
    * HCMS (35 vols, 8 layers each B-scan), [publication](https://www.sciencedirect.com/science/article/pii/S2352340918316135), [data](https://medic.rad.jhmi.edu/index.php?title=OCT_Data)
* Secondly, with the addition of custom training data of the project,( 7 layers, 224 B-scans)

## Layers
The segmentation of the custom project data was limited to the 7 layers:
* RNFL: Retina nerve fiber layer
* GCL+IPL: Ganglion cell layer and inner plexiform layer
* INL: Inner nuclear layer
* OPL: Outer plexiform layer
* ONL: Outer nuclear layer
* EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
* RPE: Retinal pigment epithelium

For HCMS, the inner and outer photoreceptot segments were combined to the ellipsoid zone.

## Voxel size
The data was exported in NIfTI format with the voxel sizes:
* HCMS: (3.87, 5.8, 123.6) µm
* DUKE_DME: (3.87, 11.33) µm
* custom data: (3.87, 5.88) µm

## Export into nnU-Net data format
Both external and internal data has to be transformed into a format, which is compatible with nnU-Net and features spatial information in form of the voxel size.
A good choice for this is the NIfTI file format.
The conversion process creates two directories, `imagesTr` and `labelsTr` inside the output directory, which can be used for the training of the nnU-Net.

### Public datasets
The public datasets can be downloaded with the links referenced above.
The conversion into this file format can be performed like this:
```bash
# HCMS dataset
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/HCMS_DATA/OCT_Manual_Delineations-2018_June_29/ -o <OUTPUT_DIR> --dataset hcms
# DUKE DME
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/DUKE_DME_DATA/duke_dme_2015_BOE_Chiu2/2015_BOE_Chiu/ -o <OUTPUT_DIR> --dataset duke_dme
```

### Internal data
The internal data is stored in h5 files.
They contain different versions of the labels, e.g. the original annotations are stored under `["labels"]["orig"]`.
The annotations can be further refined by removing stray pixels, restricting the labels to the image, and synchronizing the label IDs, so that the same label ID always corresponds to the same semantic layer.
```bash
python scripts/process_nnunet_data/nnunet_preprocess_internal_data.py -i <INPUT_DIR> -o <OUTPUT_DIR> -l edit_v3
```

## Analysis
We can see a clear difference in the performance of the nnU-Net when only trained on the public datasets and when trained on internal data.
While the public datasets always include every layer, the internal data features cases, where the lower layers of EZ, ONL, and OPL are degenerated and missing.

nnU-Net has been trained for 5 folds, which were used for inference using cross-validation
```
nnUNetv2_predict -d Dataset004_OCT-2d-all -i 20250717_input -o 20250717_seg_004-2d_all -f  0 1 2 3 4 -tr nnUNetTrainer -c 2d -p nnUNetPlans
```
