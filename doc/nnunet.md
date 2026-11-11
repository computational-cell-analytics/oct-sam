# Segment OCT B-scans with nnU-Net

This guide installs nnU-Net and the pre-trained model.
Then it segments one B-scan into the 7 retinal layers and shows the result in napari.

You need:
- Linux, macOS or Windows (WSL) with a terminal.
- About 10 GB of free disk space for the two environments and the model.
- A B-scan in TIF format. See [Input data](../README.md#input-data).

You can run each step again. A step skips the parts that are already complete.

Two environments are necessary:
- `oct-sam` for this repository. It converts the data and opens napari.
- `nnunet` for nnU-Net. It runs the network.

The commands use micromamba.
If you use conda, replace `micromamba` with `conda` in all commands, and add `-m conda` to `oct_tools.apply_nnunet`.

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

## 3. Install the `nnunet` environment

```bash
if ! micromamba run -n nnunet pip show nnunetv2 >/dev/null 2>&1; then
    micromamba create -y -n nnunet -c conda-forge python=3.12
    micromamba run -n nnunet pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    micromamba run -n nnunet pip install nnunetv2
fi
```

These commands install PyTorch for the CPU.
For an NVIDIA GPU, select the correct PyTorch build on [pytorch.org](https://pytorch.org/get-started/locally/).

**Result:** `micromamba env list` shows `nnunet`.

## 4. Set the nnU-Net model folder

nnU-Net installs its models into the folder in the environment variable `nnUNet_results`.

```bash
if [ -z "$nnUNet_results" ]; then
    echo 'export nnUNet_results="$HOME/nnUNet_results"' >> ~/.bashrc
    export nnUNet_results="$HOME/nnUNet_results"
fi
echo "$nnUNet_results"
```

**Result:** The terminal shows the model folder.
The line in `~/.bashrc` sets the variable in each new terminal.

## 5. Download and install the model

```bash
mkdir -p checkpoints
[ -f checkpoints/nnunet_model_001_oct-2d.zip ] || curl -L -o checkpoints/nnunet_model_001_oct-2d.zip https://owncloud.gwdg.de/index.php/s/ffZ4MvFWt8E5jpv/download
[ -d "$nnUNet_results/Dataset001_OCT-2d" ] || micromamba run -n nnunet nnUNetv2_install_pretrained_model_from_zip checkpoints/nnunet_model_001_oct-2d.zip
ls "$nnUNet_results"
```

The download is 345 MB.
If the download stops, delete `checkpoints/nnunet_model_001_oct-2d.zip` and run the step again.

**Result:** The folder list shows `Dataset001_OCT-2d`.

## 6. Segment a B-scan

Copy your B-scan into the folder `data`:

```bash
mkdir -p data
cp /path/to/your_bscan.tif data/bscan.tif
```

Segment it:

```bash
oct_tools.apply_nnunet -i data/bscan.tif -o data/bscan_nnunet.tif
```

- Add `--device cuda` if you have an NVIDIA GPU. The default is `cpu`.
- nnU-Net shows warnings about `nnUNet_raw` and `nnUNet_preprocessed`. The segmentation does not use these folders.
- A stack (TIF, H5 or VOL) gives a stack of segmentations, one for each B-scan.

**Result:** The terminal shows `Output: data/bscan_nnunet.tif`.

## 7. Show the result in napari

```bash
oct_tools.open_labels -i data/bscan.tif data/bscan_nnunet.tif
```

- For a stack, add `--slice <z>` to show B-scan `z`.

**Result:** napari opens.
- The layer list on the left shows the B-scan `data/bscan.tif` and the segmentation `data/bscan_nnunet.tif` on top of it.
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

- Move the mouse over a pixel. The status bar at the bottom shows its label ID.
- Use the `opacity` slider in the **layer controls** to see the B-scan under the segmentation.
- The **Label Color Map** widget on the right changes the colors. The style `check` shows the IDs 1-7 in green and all other IDs in red.

## Next steps

- Segment all TIF files of a folder: `oct_tools.apply_nnunet -i data/images -o data/segmentations`.
  The command writes one segmentation `<name>.tif` for each image `<name>.tif`, and `<name>_z<slice>.tif` for each B-scan of a stack.
  The output folder must not be the input folder.
- Measure the layer thicknesses: [measurement.md](measurement.md).
- Segment mouse OCT: [2d_mouse.md](2d_mouse.md).
- Retrain the model and details of the training: [nnunet_training.md](nnunet_training.md).
