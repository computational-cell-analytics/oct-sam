# Segment OCT B-scans with OCT-SAM

This guide installs OCT-SAM and downloads the trained model.
Then it segments one B-scan automatically, corrects the segmentation interactively in napari, and shows the result.

OCT-SAM is a fine-tuned [MedicoSAM](https://doi.org/10.1109/TMI.2025.3644811) model.
For its training, see [oct-sam_training.md](oct-sam_training.md).

You need:
- Linux, macOS or Windows (WSL) with a terminal.
- About 6 GB of free disk space for the environment and the model.
- A B-scan in TIF format. See [Input data](../README.md#input-data).

You can run each step again. A step skips the parts that are already complete.

The commands use micromamba.
If you use conda, replace `micromamba` with `conda` in all commands.

## 1. Install micromamba

Follow the [micromamba installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).

**Result:** `micromamba --version` shows the version.

## 2. Install the `oct-sam` environment

```bash
[ -d oct-sam ] || git clone https://github.com/computational-cell-analytics/oct-sam.git
cd oct-sam
if ! micromamba run -n oct-sam pip show oct-sam >/dev/null 2>&1; then
    micromamba env create -y -f environment.yaml
    micromamba run -n oct-sam pip install -e .
fi
micromamba activate oct-sam
```

**Result:** The prompt starts with `(oct-sam)`. Run all further commands in this terminal, in the folder `oct-sam`.

## 3. Download the model

```bash
mkdir -p checkpoints
[ -f checkpoints/oct-sam-V1.pt ] || curl -L -o checkpoints/oct-sam-V1.pt https://owncloud.gwdg.de/index.php/s/12FhJAc8XTNzHLA/download
ls -lh checkpoints
```

The download is 413 MB.
If the download stops, delete `checkpoints/oct-sam-V1.pt` and run the step again.

**Result:** The folder list shows `oct-sam-V1.pt`.

## 4. Segment a B-scan automatically

Copy your B-scan into the folder `data`:

```bash
mkdir -p data
cp /path/to/your_bscan.tif data/bscan.tif
```

Segment it and open napari:

```bash
oct_tools.interactive -i data/bscan.tif -o data/oct-sam --model checkpoints/oct-sam-V1.pt --precompute_segmentation
```

- For a TIF stack, add `-z <slice>` to select the B-scan, for example `-z 31`. The default is the first B-scan.
- On a CPU, the first run takes some minutes. It computes the image embedding and the automatic segmentation.

**Result:** napari opens.
- The layer `committed_objects` shows the automatic segmentation on top of the B-scan.
- Each retinal layer has its own color:

| ID | Layer | Name |
|---|---|---|
| 1 | RNFL | Retinal nerve fiber layer |
| 2 | GCIPL | Ganglion cell layer and inner plexiform layer |
| 3 | INL | Inner nuclear layer |
| 4 | OPL | Outer plexiform layer |
| 5 | ONL | Outer nuclear layer |
| 6 | EZ | Ellipsoid zone |
| 7 | RPE | Retinal pigment epithelium |

- The output folder contains the automatic segmentation `data/oct-sam/bscan_0.tif`.
  The number is the index of the B-scan in the TIF file.

## 5. Correct a layer interactively

Remove the incorrect layer:
1. Select the layer `committed_objects` in the layer list.
2. In the **layer controls**, select the fill tool (bucket) and set `label` to `0`.
3. Click on the incorrect layer. The layer disappears.

Segment the layer again with point prompts:
1. Select the layer `point_prompts` and the tool `Add points`.
2. Click on the retinal layer at some positions along its length. Each click adds a positive point.
3. Press `T` to change to negative points. Click on the neighboring layers. Press `T` again to change back.
4. Press `S` (**Segment Object**). The layer `current_object` shows the result.
   Add or remove points and press `S` again until the result is correct.
5. Press `C` (**Commit**). The object moves into `committed_objects` and gets a new label ID.
   The measurement identifies the layers by their vertical order, so the new ID is not a problem.

Press `Shift+C` (**Clear Annotations**) to remove all points and `current_object`.

## 6. Save the segmentation

Press `N` (**Next Image**).

**Result:** napari saves the corrected segmentation to `data/oct-sam/bscan_0.tif`.
For the last B-scan, napari shows a message. Confirm it, and napari closes.

The command of step 4 opens the saved segmentation again.
To compute a new automatic segmentation, delete `data/oct-sam/bscan_0.tif` first.

The window of step 4 also contains the measurement widgets. [measurement.md](measurement.md) describes them.

## 7. Show the result in napari

```bash
oct_tools.open_labels -i data/oct-sam/bscans/bscan_0.tif data/oct-sam/bscan_0.tif
```

The folder `data/oct-sam/bscans` contains a copy of each segmented B-scan.

**Result:** napari opens with the B-scan and the corrected segmentation on top of it.
- Move the mouse over a pixel. The status bar at the bottom shows its label ID.
- The **Label Color Map** widget on the right changes the colors. The style `check` shows the IDs 1-7 in green and all other IDs in red.

## Next steps

- Measure the layer thicknesses: [measurement.md](measurement.md).
- Segment many B-scans without napari: see [Batch segmentation](#batch-segmentation).

## Batch segmentation

`oct_tools.apply_sam` segments an image file, or all H5 and TIF files of a folder, without napari:

```bash
oct_tools.apply_sam -i data/bscan.tif -m checkpoints/oct-sam-V1.pt -o data/oct-sam-batch
```

The command writes one segmentation `<name>.tif` for each image.
For a TIF stack `(z, y, x)`, it writes one segmentation `<name>_<z>.tif` for each B-scan.
`-z` selects B-scans, for example `-z 0 24`. Without `-z`, the command segments all B-scans.
Add `--output_extension h5` to write H5 files. Add `-f` to replace existing segmentations.

The segmentation has two phases:
1. The network segments the B-scan. The command derives point prompts from this segmentation.
2. The network segments the B-scan again with these prompts.

`--no_prompts` skips the second phase.

Then the command post-processes the segmentation.
`--postprocess_functions` selects the steps and their order:

| Step | Effect |
|---|---|
| `merge_horizontal` | Merges the separate parts of a layer along the B-scan. |
| `filter_thin` | Removes segments that are thinner than 5 pixels. |
| `filter_fragments` | Removes parts of a layer that are much smaller than its largest part. |
| `fill_gaps` | Fills holes that do not touch the background above or below the retina, with a watershed. |
| `assign_layer_id` | Gives the layers the IDs 1-7 from their vertical order. It must be the last step. |

The default is `merge_horizontal filter_thin filter_fragments assign_layer_id`.
Use `--postprocess_functions none` for no post-processing.
