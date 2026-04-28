# Training data

## Public datasets

### HCMS - Healthy Control - Multiple Sclerosis

* [publication](https://doi.org/10.1016/j.dib.2018.12.073)
* [data](https://medic.rad.jhmi.edu/index.php?title=OCT_Data)
* 35 volunteers:
    * 14 healthy controls (HC)
    * 21 with a diagnosis of multiple sclerosis (MS)
* 49 B-scans per volume, 8 layers in each B-scan
* 3 datasets (MS 14, MS 16, MS17) were omitted because no control points were found
* Featured layers:
    * RNFL: Retina nerve fiber layer
    * GCL+IPL: Ganglion cell layer and inner plexiform layer
    * INL: Inner nuclear layer
    * OPL: Outer plexiform layer
    * ONL: Outer nuclear layer
    * IS: Inner photoreceptor segments
    * OS: Outer photoreceptor segments
    * RPE: Retinal pigment epithelium
* Resolution (mean and standard deviation):
    * Lateral resolution (between A-scans) is 5.8 µm (±0.2)
    * Axial resolution (between two pixels in an A-scan) is 3.9 µm (±0.0)
    * Through-plane distance (slice separation) is 123.6 µm (±3.6) between images
    * Imaging area of approximately 6 x 6 mm²

#### Duke DME

* [publication](https://doi.org/10.1117/12.2654210)
* [data](https://people.duke.edu/~sf59/software.html)
* 10 subjects, sparsely labeled
* 110 B-scans, 7–8 layers + fluids
* Featured layers:
    * NFL: Nerve fiber layer
    * GCL-IPL: Ganglion cell layer - Inner plexiform layer
    * INL: Inner nuclear layer
    * OPL: Outer plexiform layer
    * ONL-ISM: Outer nuclear layer - Inner segment myeloid
    * ISE: Inner segment ellipsoid
    * OS-RPE: Outer segment - Retinal pigment epithelium
* Resolution:
    * Lateral resolution: ranging from 11.07 - 11.59 µm/pixel
    * Axial resolution: 3.87 µm/pixel
    * Azimuthal resolution: ranging from 118 - 128 µm/pixel

### Download
The public datasets can be downloaded with the links referenced above.

### Export into nnU-Net data format
Both external and internal data has to be transformed into a format, which is compatible with nnU-Net and features spatial information in form of the voxel size.
A good choice for this is the NIfTI file format.
The conversion process creates two directories, `imagesTr` and `labelsTr` inside the output directory, which can be used for the training of the nnU-Net.

The conversion into the file format can be performed like this:
```bash
# HCMS dataset
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/HCMS_DATA/OCT_Manual_Delineations-2018_June_29/ -o <OUTPUT_DIR> --dataset hcms
# DUKE DME
python scripts/process_nnunet_data/nnunet_preprocess_external_data.py -i /path/to/DUKE_DME_DATA/duke_dme_2015_BOE_Chiu2/2015_BOE_Chiu/ -o <OUTPUT_DIR> --dataset duke_dme
```

#### OCT-SAM

## Custom Data
The internal data is stored in H5 files.
They contain different versions of the labels, e.g. the original annotations are stored under `["labels"]["orig"]`.
The annotations can be further refined by removing stray pixels, restricting the labels to the image, and synchronizing the label IDs, so that the same label ID always corresponds to the same semantic layer.
```bash
python scripts/process_nnunet_data/nnunet_preprocess_internal_data.py -i <INPUT_DIR> -o <OUTPUT_DIR> -l edit_v3
```

## Layers
The segmentation data was limited to the 7 layers:
* RNFL: Retina nerve fiber layer
* GCL+IPL: Ganglion cell layer and inner plexiform layer
* INL: Inner nuclear layer
* OPL: Outer plexiform layer
* ONL: Outer nuclear layer
* EZ: Ellipsoid zone (Inner photoreceptor segments and Outer photoreceptor segments)
* RPE: Retinal pigment epithelium

For HCMS, the inner and outer photoreceptor segments were combined to the ellipsoid zone.

### Voxel size
The data was exported in NIfTI format with the voxel sizes:
* HCMS: (3.87, 5.8, 123.6) µm
* DUKE_DME: (3.87, 11.33) µm
* custom data: (3.87, 5.88) µm
