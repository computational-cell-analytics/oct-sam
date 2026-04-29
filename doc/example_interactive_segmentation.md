# Example for the interactive segmentation using OCT-SAM

This script will give an example for the interactive segmentation of an OCT B-scan.


## Download the OCT-SAM model

Download the OCT-SAM model `oct-sam-V1.pt` from ownCloud:
[Download from ownCloud](https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA)

## Install `oct_tools` in your environment
Run `pip install -e .` to install functionalities for the command line interface (CLI).

## Run an interactive segmentation

To apply an OCT-SAM model and measure the thicknesses of retinal layers at the same time, use the `oct_tools.interactive` function:
```bash
oct_tools.interactive --model /path/to/oct-sam-model.pt --precompute_segmentation --output /path/to/output_folder/ --input /path/to/input_image.h5
```

## Run a retrospective measurement
It is also possible to measure retinal layers after the segmentation is completed by using the `oct_tools.measure` function:
```bash
oct_tools.measure -i /path/to/input_image.tif -s /path/to/input_segmentation.tif -o /path/to/output_folder/
```

## Analyse retinal layers with napari

Napari uses different layers, which can be individually activated and manipulated by clicking on the respective layer.
A list of all available layers titled `layer list` can be found on the left side of the napari GUI.

### Measure the central foveal thickness and ETDRS-like areas

Go to the `fovea reference point` layer on the left side of the GUI and activate it.
Go to the `layer controls` section and activate `Select points`.
Move the white dot to the foveal point or remove it if such a point is not present in the slice.

Click the `Measure` button on the right side of the interface to create a measurement table which features the central foveal thickness (CFT)


# Calculate metrics for a segmentation
```bash
oct_tools.metrics --input /path/to/input_seg.tif
```

## Apply OCT-SAM network
The OCT-SAM model can also be applied non-interactively.
```bash
oct_tools.apply_sam --model /path/to/oct-sam-model.pt --postprocess_functions merge_horizontal filter_thin --output /path/to/output_folder/ --input /path/to/input_data.h5
```