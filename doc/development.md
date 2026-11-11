# Development

This page is for developers of the `oct_tools` package.
For the installation, see [README.md](../README.md#installation). `pip install -e .` installs the package in editable mode, so a code change has an effect immediately.

## Tests and lint

```bash
python -m unittest discover -s test -v
flake8 .
```

- Use `discover`. The standard library also has a package `test`, so `python -m unittest test.test_x` imports the wrong package.
- The tests make small synthetic B-scans in memory. They do not need data from the cluster, and they run without a display.
- There is no CI. For behavior that the tests do not cover, run the command on real data and check the result in napari.

## Package layout

```
oct_tools/
├── cli.py                      # All console_scripts entry points (argparse only)
├── layer_information.py        # Layer names, label IDs 1-7, napari color maps
├── apply_oct_sam.py            # Batch OCT-SAM inference
├── precompute_segmentation.py  # SAM prompt derivation, batched inference, mask NMS
├── interactive_segmentation.py # napari µSAM annotator (oct_tools.interactive)
├── measure_segmentation.py     # napari measurement viewer (oct_tools.measure)
├── apply_nnunet.py             # TIF/H5 <-> NIfTI, nnUNetv2_predict and nnUNetv2_train subprocesses
├── postprocessing.py           # Post-processing steps of a segmentation
├── refine_annotations.py       # Clean-up of manual annotations, layer IDs, binarization
├── data_reader.py              # Reader for all TIF, H5 and VOL input
├── heyex_vol.py                # Heidelberg Heyex VOL reader, VOL -> TIF (oct_tools.convert_vol)
├── metric_utils.py             # Thickness, length, area and ETDRS-like metrics
├── eval_segmentation.py        # Precision, recall, F1 score and symmetric best Dice
├── train_utils.py              # OCT-SAM training, model export, split generation
├── analysis/                   # Evaluation of experiments
└── napari_widgets/             # Widgets of both napari apps
```

`oct_tools/analysis/` and `oct_tools/napari_widgets/` have no `__init__.py`. Python imports them as namespace packages.
`setup.py` uses `find_namespace_packages`, so a non-editable install includes them.

Each command of `setup.py` has one function in `oct_tools/cli.py`.
The function makes an argparse parser and calls the implementation module.
Register a new command in `setup.py`.

nnU-Net is not a Python dependency. `oct_tools/apply_nnunet.py` runs the nnU-Net commands in the separate `nnunet` environment with `subprocess`.

## Data flow

1. Heyex VOL tomogram -> `oct_tools.convert_vol` -> TIF.
2. Manual annotation in napari -> `scripts/process_train_data/training_data_cycles.py` -> H5 with `image` and `labels/original`.
3. `scripts/process_train_data/edit_annotations.py` -> `labels/edit_v1`, `edit_v2`, `edit_v3`. `edit_v3` is the training label.
4. Conversion for each model:
   - nnU-Net: `scripts/process_nnunet_data/nnunet_preprocess_{external,internal}_data.py` -> NIfTI `imagesTr` and `labelsTr`.
   - OCT-SAM: `scripts/process_train_data/preprocess_external_data.py` -> H5. UMG-RP H5 files need no conversion.
5. Splits: `doc/train_splits/*.json` and `scripts/process_nnunet_data/create_retrain_data.py`.
6. Training: `scripts/training/` for OCT-SAM. `nnUNetv2_plan_and_preprocess` and `nnUNetv2_train` for nnU-Net.
7. Inference: `oct_tools.apply_sam` or `oct_tools.apply_nnunet`.
8. Evaluation: `oct_tools.eval_segmentation`, `scripts/oct-sam/eval_sam.py`, `scripts/oct-sam/eval_iterative_prompting.py`.
9. Clinical measurement: `oct_tools.measure` or `oct_tools.metrics`, then `scripts/compare_seg_to_thickness_measurement.py`.
10. Figures: `results/figures/*.py`.

## Scripts

| Path | Content |
|---|---|
| `scripts/process_train_data/` | Makes and checks the H5 training data. See [training_data.md](training_data.md). |
| `scripts/process_nnunet_data/` | Converts the data for nnU-Net and copies split subsets. See [nnunet_training.md](nnunet_training.md). |
| `scripts/training/` | Trains OCT-SAM. See [oct-sam_training.md](oct-sam_training.md). |
| `scripts/oct-sam/` | Evaluates OCT-SAM: `eval_sam.py` for a folder of B-scans, `eval_iterative_prompting.py` for prompt-based correction. |
| `scripts/mouse/annotate.py` | Corrects mouse segmentations in napari. See [2d_mouse.md](2d_mouse.md). |
| `scripts/compare_seg_to_thickness_measurement.py` | Compares nnU-Net thicknesses with manual measurements. |

## Results

| Path | Content |
|---|---|
| `results/model_accuracy.xlsx` | Summary of the accuracy of all models |
| `results/model_accuracy_validation/` | Accuracy of each B-scan of the validation dataset `20250717`, one file for each model |
| `results/iterative_prompts/` | Accuracy of OCT-SAM after each correction step |
| `results/clinical_validation/` | Manual and nnU-Net thickness measurements |
| `results/figures/` | Scripts for the figures |

The file names use internal model names. See [oct-sam_training.md](oct-sam_training.md#for-maintainers).

## Conventions

### Axis order

- A 2D B-scan or segmentation is `(y, x)`: axial (depth) and lateral. Each column is one A-scan.
- A stack is `(z, y, x)`. `z` is the B-scan index.
- napari points use the same order. A fovea position or reference position on the command line is an `x` value (a column).

### Voxel size

`VOXEL_SIZE` in `oct_tools/metric_utils.py` is in µm, in the order `(z, y, x)`.
2D code uses `VOXEL_SIZE[1:]`, so in a `spacing` argument:
- `spacing[0]` is the axial pixel size. A thickness is the number of pixels times `spacing[0]`.
- `spacing[1]` is the lateral pixel size. The ETDRS ring widths use `spacing[1]`.

`oct_tools.metrics -v` uses the same order. Do not reverse it.

### Labels

- `0` is the background. `1` to `7` are the retinal layers. See [README.md](../README.md#retinal-layers).
- `oct_tools/layer_information.py` is the single source for the layer IDs and colors.
- Raw SAM output has arbitrary instance IDs. The post-processing step `assign_layer_id` changes them to 1-7.
- IDs from `8` to `255` are not valid layers. napari shows them in vivid colors. The color style `check` shows 1-7 in green and all other IDs in red.
- A binary label has `0` for the background and `1` for the retina.
- Label arrays are written as `uint32`.

### Files

| Format | Use |
|---|---|
| TIF | B-scans and segmentations |
| H5 | Training data and annotations. Keys: `image`, `labels/original`, `labels/edit_v1` to `edit_v3`, `segmentation`, `prompts`. |
| NIfTI `.nii.gz` | nnU-Net data |
| Heyex `.vol` | Raw Spectralis tomograms |
| TSV, XLSX | Measurement tables |
| JSON | Evaluation results and splits |

- Read every TIF, H5 or VOL input with `oct_tools.data_reader`. `read_volume` returns `(z, y, x)`.

- UMG-RP H5 files: `RP<id>_<year>_<eye>_z<slice>.h5`.
- nnU-Net case names have a dataset prefix: `11` and `12` for HCMS, `20` for Duke DME, `30` for UMG-RP. `RP012_2023_OD_z08.h5` becomes `oct_30012_2023_OD_008`.
- nnU-Net image files end with `_0000.nii.gz`. Label files do not.
- `oct_tools.interactive` names each B-scan `<image name>_<z>`. In the output folder, the segmentation is `<name>.tif`, the B-scan is `bscans/<name>.tif` and the embedding is `embeddings/<name>.zarr`.

### Code style

- Google-style docstrings (`Args:`, `Returns:`) on public functions.
- Type hints on public functions with `typing.List`, `Optional`, `Tuple` and `Union`.
- Commit messages: a short capitalized phrase without a period, with the PR number if there is one. Example: `Retrain nnunet (#32)`.
