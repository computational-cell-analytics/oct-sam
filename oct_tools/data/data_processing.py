import json
import os
import re
import shutil
from glob import glob
from typing import Dict, Iterable, Iterator, List, NamedTuple, Optional, Sequence, Tuple

import h5py
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from oct_tools.data.data_reader import nifti_affine, read_bscan, read_nifti, read_volume
from oct_tools.data.refine_annotations import binarize_layer_label

DATASETS = ("hcms", "duke_dme", "umg_rp")
# Pixel spacing (y, x) in um that the conversion writes into the NIfTI files. See doc/training_data.md.
SPACING = {
    "hcms": (3.87, 5.8),
    "duke_dme": (3.87, 11.33),
    "umg_rp": (3.87, 5.88),
}
# These B-scans miss layers.
UMG_RP_EXCLUDED = ("RP122_2025_OS_z08", "RP041_2025_OD_z10")

# Prefix of the legacy nnU-Net names of HCMS: hc = healthy control, ms = multiple sclerosis.
_HCMS_PREFIXES = {"hc": "11", "ms": "12"}
_SOURCES = {"hcms": "hcms", "duke": "duke_dme", "umgrp": "umg_rp"}


class Sample(NamedTuple):
    """One annotated B-scan.

    `name` is the standard name, e.g. "hcms_hc_01_z00". The legacy names are the names of the earlier conversions
    for nnU-Net (e.g. "oct_11001_000") and for OCT-SAM (e.g. "hc01_spectralis_macula_v1_s1_R_000").
    See doc/training_data.md. `image` and `label` have the shape (y, x).
    """
    name: str
    legacy_nnunet_name: str
    legacy_h5_name: str
    image: np.ndarray
    label: np.ndarray


def _hcms_name(group: str, number: str, z: str) -> str:
    return f"hcms_{group}_{int(number):02}_z{int(z):02}"


def _duke_name(number: str, z: str) -> str:
    return f"duke_{int(number):02}_z{int(z):02}"


def _umg_rp_name(patient: str, year: str, eye: str, z: str) -> str:
    return f"umgrp_{int(patient):03}_{year}_{eye}_z{int(z):02}"


_LEGACY_HCMS_GROUPS = {prefix: group for group, prefix in _HCMS_PREFIXES.items()}
_NAME_PATTERNS = (
    (r"hcms_(hc|ms)_(\d+)_z(\d+)", _hcms_name),
    (r"(hc|ms)(\d+)_spectralis_macula_v1_s1_R_(\d+)", _hcms_name),
    (r"oct_(1[12])(\d{3})_(\d{3})", lambda prefix, number, z: _hcms_name(_LEGACY_HCMS_GROUPS[prefix], number, z)),
    (r"duke_(\d+)_z(\d+)", _duke_name),
    # Dataset 013 (2026-09-02) used the form oct_20001_010.
    (r"oct_20(\d{3})_?(\d{3})", _duke_name),
    (r"umgrp_(\d+)_(\d{4})_(O[DS])_z(\d+)", _umg_rp_name),
    (r"RP(\d+)_(\d{4})_(O[DS])_z(\d+)", _umg_rp_name),
    (r"oct_30(\d{3})_(\d{4})_(O[DS])_(\d{3})", _umg_rp_name),
)


def standard_name(name: str) -> str:
    """Return the standard name of a B-scan, e.g. "duke_01_z10" for "oct_20001010".

    The function accepts the standard names, the legacy nnU-Net names and the legacy H5 names of HCMS and
    UMG-RP. A legacy H5 name of Duke DME, e.g. "Subject_01_z000", counts only the annotated B-scans, so the
    function cannot convert it. See doc/training_data.md.

    Raises:
        ValueError: If the name has no known form.
    """
    for pattern, build in _NAME_PATTERNS:
        match = re.fullmatch(pattern, name)
        if match is not None:
            return build(*match.groups())
    raise ValueError(f"{name} is not a known B-scan name such as hcms_hc_01_z00, duke_01_z10 or umgrp_012_2023_OD_z08.")


def case_source(name: str) -> str:
    """Return the dataset of a B-scan name: "hcms", "duke_dme" or "umg_rp"."""
    return _SOURCES[standard_name(name).split("_")[0]]


def legacy_nnunet_name(name: str) -> str:
    """Return the legacy nnU-Net name of a B-scan, e.g. "oct_20001010" for "duke_01_z10"."""
    source, *fields = standard_name(name).split("_")
    z = int(fields[-1][1:])
    if source == "hcms":
        return f"oct_{_HCMS_PREFIXES[fields[0]]}{int(fields[1]):03}_{z:03}"
    if source == "duke":
        return f"oct_20{int(fields[0]):03}{z:03}"
    patient, year, eye = fields[:3]
    return f"oct_30{patient}_{year}_{eye}_{z:03}"


def umg_rp_source_name(name: str) -> str:
    """Return the name of the UMG-RP H5 file of a B-scan, e.g. "RP012_2023_OD_z08" for "umgrp_012_2023_OD_z08"."""
    standard = standard_name(name)
    if not standard.startswith("umgrp_"):
        raise ValueError(f"{name} is not a UMG-RP name.")
    return "RP" + standard[len("umgrp_"):]


def merge_photoreceptor_segments(label: np.ndarray) -> np.ndarray:
    """Merge the 8 HCMS layers into the 7 layers of the models.

    The outer photoreceptor segments (7) join the inner segments (6) as the ellipsoid zone, and the RPE moves
    from 8 to 7.
    """
    if label.max() != 8:
        raise ValueError(f"An HCMS label must have the RPE as ID 8, but its highest ID is {label.max()}.")
    label = label.copy()
    label[label == 7] = 6
    label[label == 8] = 7
    return label


def _surfaces_to_label(control_pts: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    n_slices, n_surfaces = control_pts.shape
    assert n_slices == shape[0]

    # Use only the surfaces that contain points.
    nonempty = np.array(
        [[control_pts[i, j].size > 0 for j in range(n_surfaces)]
         for i in range(n_slices)]
    )
    used_surfaces = np.where(nonempty.any(axis=0))[0]

    seg = np.zeros(shape, dtype="uint8")
    _, height, width = shape

    # The control points are 1-based.
    x_grid = np.arange(1, width + 1, dtype=np.float32)

    n_used = len(used_surfaces)
    surface_y = np.empty((n_used, width), dtype=np.float32)

    for s in range(n_slices):
        for idx, j in enumerate(used_surfaces):
            pts = control_pts[s, j]
            if pts.size == 0:
                surface_y[idx, :] = np.nan
                continue

            xs = pts[:, 0].astype(np.float32)
            ys = pts[:, 1].astype(np.float32)

            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]

            # Outside the annotated range, use the end points.
            surface_y[idx, :] = np.interp(x_grid, xs, ys, left=ys[0], right=ys[-1])

        # The surfaces are ordered inner to outer (small y to large y). Each pair of surfaces gives one band.
        for b in range(n_used - 1):
            y_top = surface_y[b, :] - 1.0
            y_bot = surface_y[b + 1, :] - 1.0

            y_top = np.clip(np.floor(y_top).astype(int), 0, height - 1)
            y_bot = np.clip(np.ceil(y_bot).astype(int), 0, height - 1)

            for x in range(width):
                t = y_top[x]
                btm = y_bot[x]
                if btm <= t:
                    continue
                seg[s, t:btm, x] = b + 1

    return seg


def read_hcms(input_dir: str, merge_is_os: bool = True) -> Iterator[Sample]:
    """Read the annotated B-scans of the HCMS dataset.

    Publication: https://doi.org/10.1016/j.dib.2018.12.073
    Data: https://medic.rad.jhmi.edu/index.php?title=OCT_Data

    Args:
        input_dir: Root folder of the extracted download, with the subfolders "vol" and "delineation".
        merge_is_os: Merge the inner and outer photoreceptor segments into the ellipsoid zone.
            Without the merge, the label keeps the 8 annotated layers.

    Yields:
        One sample for each B-scan.
    """
    tomograms = sorted(glob(os.path.join(input_dir, "vol", "*.vol")))
    for tomo in tqdm(tomograms, desc="HCMS volumes"):
        name = os.path.splitext(os.path.basename(tomo))[0]
        if name[:2] not in _HCMS_PREFIXES:
            raise ValueError(f"The name of {tomo} must start with one of {sorted(_HCMS_PREFIXES)}.")
        try:
            control_pts = loadmat(os.path.join(input_dir, "delineation", f"{name}.mat"))["control_pts"]
        except KeyError:
            print("Could not find control points for", tomo)
            continue

        data = read_volume(tomo)
        labels = _surfaces_to_label(control_pts, data.shape)
        for z in range(len(data)):
            try:
                label = merge_photoreceptor_segments(labels[z]) if merge_is_os else labels[z]
            except ValueError as error:
                raise ValueError(f"{tomo}, B-scan {z}: {error}") from error
            standard = _hcms_name(name[:2], name[2:4], z)
            yield Sample(standard, legacy_nnunet_name(standard), f"{name}_{z:03}", data[z], label)


def _load_duke_data(data: dict) -> Tuple[np.ndarray, np.ndarray]:
    images = data["images"]  # (y, x, z)
    H, W, N = images.shape
    images_bhw = np.transpose(images, (2, 0, 1))

    layers = data["manualLayers1"]
    L, W_l, N_l = layers.shape
    assert W_l == W and N_l == N

    labels = np.zeros((N, H, W), np.uint8)

    xs_full = np.arange(W, dtype=np.float64)

    for b in range(N):
        if np.all(np.isnan(layers[:, :, b])):
            continue

        surf_y = np.full((L, W), np.nan, np.float64)

        for s in range(L):
            y = layers[s, :, b]
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue

            xs = xs_full[valid]
            ys = y[valid]

            # Interpolate only inside the annotated range.
            mask = (xs_full >= xs.min()) & (xs_full <= xs.max())
            surf_y[s, mask] = np.interp(xs_full[mask], xs, ys)

        # Use only the columns in which all surfaces are defined.
        cols = np.where(~np.isnan(surf_y).any(axis=0))[0]
        if len(cols) == 0:
            continue

        # The surfaces are 1-based.
        surf_pix = np.round(surf_y[:, cols] - 1).astype(int)
        surf_pix = np.clip(surf_pix, 0, H - 1)

        for band in range(L - 1):
            y0 = surf_pix[band]
            y1 = surf_pix[band + 1]
            top = np.minimum(y0, y1)
            bot = np.maximum(y0, y1)
            for i, x in enumerate(cols):
                if bot[i] > top[i]:
                    labels[b, top[i]:bot[i] + 1, x] = band + 1

    return images_bhw, labels


def read_duke_dme(input_dir: str) -> Iterator[Sample]:
    """Read the annotated B-scans of the Duke DME dataset.

    Publication: https://doi.org/10.1364/BOE.6.001172
    Data: https://people.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm

    The label is the first manual annotation (`manualLayers1`). Only the annotated B-scans are read,
    and each one is cropped to the annotated columns.

    Args:
        input_dir: Root folder of the extracted download with the MAT files.

    Yields:
        One sample for each annotated B-scan. `name` and `legacy_nnunet_name` have the index of the B-scan in
        the volume, `legacy_h5_name` the index among the annotated B-scans.
    """
    files = sorted(glob(os.path.join(input_dir, "**", "*.mat"), recursive=True))
    for path in tqdm(files, desc="Duke DME volumes"):
        name = os.path.splitext(os.path.basename(path))[0]
        images, labels = _load_duke_data(loadmat(path, struct_as_record=False, squeeze_me=True))
        annotated = [z for z in range(len(labels)) if labels[z].max() > 0]
        for index, z in enumerate(annotated):
            columns = np.where(labels[z] != 0)[1]
            crop = np.s_[:, columns.min():columns.max() + 1]
            standard = _duke_name(name[-2:], z)
            yield Sample(
                standard, legacy_nnunet_name(standard), f"{name}_z{index:03}", images[z][crop], labels[z][crop]
            )


def read_umg_rp(input_dir: str, label_key: str = "edit_v3") -> Iterator[Sample]:
    """Read the UMG-RP B-scans of a folder of H5 files.

    The file names must have the form "RP<id>_<year>_<eye>_z<slice>.h5", e.g. RP012_2023_OD_z08.h5.
    The function skips the B-scans in `UMG_RP_EXCLUDED` and the files without the label.

    Args:
        input_dir: Folder with the H5 files.
        label_key: Label version in the H5 group "labels".

    Yields:
        One sample for each B-scan, with image and label as uint8.
    """
    paths = sorted(glob(os.path.join(input_dir, "*.h5")))
    # Convert all names first, so that an unknown name stops the conversion before it writes a file.
    source_names = [umg_rp_source_name(os.path.splitext(os.path.basename(path))[0]) for path in paths]
    for path, source_name in tqdm(zip(paths, source_names), total=len(paths), desc="UMG-RP B-scans"):
        if source_name in UMG_RP_EXCLUDED:
            print(f"Skipping {source_name}.")
            continue
        try:
            label = read_bscan(path, keys=(f"labels/{label_key}",))
        except KeyError:
            print(f"Skipping {path}. Label key {label_key} does not exist.")
            continue
        image = read_bscan(path)
        standard = standard_name(source_name)
        yield Sample(
            standard, legacy_nnunet_name(standard), source_name, image.astype(np.uint8), label.astype(np.uint8)
        )


def read_dataset(
    dataset: str,
    input_dir: str,
    label_key: str = "edit_v3",
    merge_is_os: bool = True,
) -> Iterator[Sample]:
    """Read the B-scans of one dataset.

    Args:
        dataset: "hcms", "duke_dme" or "umg_rp".
        input_dir: Root folder of the dataset.
        label_key: Label version of the UMG-RP data.
        merge_is_os: Merge the HCMS photoreceptor segments. See `read_hcms`.

    Returns:
        Iterator over the samples.
    """
    if dataset == "hcms":
        return read_hcms(input_dir, merge_is_os=merge_is_os)
    if dataset == "duke_dme":
        return read_duke_dme(input_dir)
    if dataset == "umg_rp":
        return read_umg_rp(input_dir, label_key=label_key)
    raise ValueError(f"Unknown dataset {dataset}. Choose one of {DATASETS}.")


def write_nnunet_case(
    output_dir: str,
    case_name: str,
    image: np.ndarray,
    label: np.ndarray,
    spacing: Tuple[float, float],
) -> None:
    """Write one B-scan as `imagesTr/<case_name>_0000.nii.gz` and `labelsTr/<case_name>.nii.gz`."""
    affine = nifti_affine(spacing)
    image_dir = os.path.join(output_dir, "imagesTr")
    label_dir = os.path.join(output_dir, "labelsTr")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    # nnU-Net needs the channel suffix on images and forbids it on labels.
    nib.save(nib.Nifti1Image(image, affine), os.path.join(image_dir, f"{case_name}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(label, affine), os.path.join(label_dir, f"{case_name}.nii.gz"))


def write_h5_sample(path: str, image: np.ndarray, label: np.ndarray, label_key: str = "masks") -> None:
    """Write one B-scan as an H5 file with the datasets "image" and `label_key`.

    An existing file is replaced only if it holds no other datasets. This protects annotated H5 files.

    Raises:
        ValueError: If `path` exists and holds other datasets.
    """
    if os.path.exists(path):
        with h5py.File(path, "r") as f:
            other = sorted(set(f.keys()) - {"image", label_key.split("/")[0]})
        if other:
            raise ValueError(f"{path} exists and holds the datasets {other}. Choose another output folder.")
    with h5py.File(path, "w") as f:
        f.create_dataset("image", data=image, compression="gzip")
        f.create_dataset(label_key, data=label, compression="gzip")


def prepare_dataset(
    samples: Iterable[Sample],
    output_dir: str,
    output_format: str,
    spacing: Tuple[float, float],
    binary: bool = False,
    legacy_names: bool = False,
) -> int:
    """Write training data for nnU-Net or OCT-SAM.

    Args:
        samples: The B-scans, e.g. from `read_dataset`.
        output_dir: Output folder.
        output_format: "nnunet" writes NIfTI files into `imagesTr` and `labelsTr`.
            "h5" writes one H5 file for each B-scan, with the label in the dataset "masks".
        spacing: Pixel spacing (y, x) in um for the NIfTI files.
        binary: Merge all retinal layers into the label 1.
        legacy_names: Use the legacy names of the output format instead of the standard names.

    Returns:
        The number of B-scans.

    Raises:
        ValueError: If `output_dir` holds a B-scan with another name, or if two samples have the same name.
            One folder must not mix standard names and legacy names, because then it holds B-scans twice.
    """
    if output_format not in ("nnunet", "h5"):
        raise ValueError(f"Unknown output format {output_format}. Choose 'nnunet' or 'h5'.")
    nnunet = output_format == "nnunet"
    os.makedirs(output_dir, exist_ok=True)
    if nnunet:
        existing = _case_index([os.path.join(output_dir, "imagesTr"), os.path.join(output_dir, "labelsTr")], ".nii.gz")
    else:
        existing = _case_index([output_dir], ".h5")

    written = set()
    for sample in samples:
        if nnunet:
            name = sample.legacy_nnunet_name if legacy_names else sample.name
        else:
            name = sample.legacy_h5_name if legacy_names else sample.name
        if name in written:
            raise ValueError(f"Two B-scans have the name {name}.")
        # The legacy H5 name of Duke DME has no standard name, so check every name of the sample.
        for alias in (sample.name, sample.legacy_nnunet_name, sample.legacy_h5_name):
            other = existing.get(_match_key(alias), name)
            if other != name:
                raise ValueError(f"{output_dir} holds the B-scan {name} with the name {other}. Use a new folder.")

        label = binarize_layer_label(sample.label) if binary else sample.label
        if nnunet:
            write_nnunet_case(output_dir, name, sample.image, label, spacing)
        else:
            write_h5_sample(os.path.join(output_dir, f"{name}.h5"), sample.image, label)
        written.add(name)
    return len(written)


def read_split(json_file: str, fold: int = 0) -> Tuple[List[str], List[str]]:
    """Read the training and the validation names of one fold of a split file in `doc/train_splits`."""
    with open(json_file) as f:
        folds = json.load(f)
    if not 0 <= fold < len(folds):
        raise ValueError(f"{json_file} has {len(folds)} folds. Fold {fold} does not exist.")
    return folds[fold]["train"], folds[fold]["val"]


def _case_of_file(filename: str) -> str:
    stem = filename[:-len(".nii.gz")] if filename.endswith(".nii.gz") else os.path.splitext(filename)[0]
    return stem[:-len("_0000")] if stem.endswith("_0000") else stem


def _match_key(name: str) -> str:
    try:
        return standard_name(name)
    except ValueError:
        return name


def _case_index(folders: Sequence[str], extension: str = "") -> Dict[str, str]:
    """Map the match key of each case in the folders to the case name. A missing folder has no cases.

    Raises:
        ValueError: If the folders hold one B-scan with two names.
    """
    cases = {}
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for entry in sorted(os.scandir(folder), key=lambda e: e.name):
            if entry.is_file() and entry.name.endswith(extension):
                case = _case_of_file(entry.name)
                other = cases.setdefault(_match_key(case), case)
                if other != case:
                    raise ValueError(f"{folder} holds the B-scan {other} also with the name {case}.")
    return cases


def copy_cases(names: Sequence[str], input_dir: str, output_dir: str) -> List[str]:
    """Copy the files of the given cases.

    A file belongs to a case if its name without the extension and without the nnU-Net channel suffix
    "_0000" is the case name, or another name of the same B-scan (see `standard_name`). For example,
    `duke_01_z10_0000.nii.gz`, `duke_01_z10.nii.gz` and `oct_20001010.nii.gz` belong to `oct_20001010`.
    The copies keep the file names.

    Args:
        names: Case names, e.g. from `read_split`.
        input_dir: Folder with the files.
        output_dir: Folder for the copies.

    Returns:
        The names that have no file in `input_dir`.

    Raises:
        ValueError: If `input_dir` holds one B-scan with two names.
    """
    cases = _case_index([input_dir])
    wanted = {_match_key(name): name for name in names}
    os.makedirs(output_dir, exist_ok=True)
    for entry in sorted(os.scandir(input_dir), key=lambda e: e.name):
        if entry.is_file() and _match_key(_case_of_file(entry.name)) in wanted:
            shutil.copy2(entry.path, os.path.join(output_dir, entry.name))
    return sorted(name for key, name in wanted.items() if key not in cases)


def split_nnunet_dataset(input_dir: str, output_dir: str, json_file: str, fold: int = 0) -> List[str]:
    """Copy the training and the validation cases of a split into a new nnU-Net dataset.

    The function also writes `splits_final.json` into `output_dir`: all folds of the split file, with the case
    names of the new dataset. nnU-Net reads this file from `$nnUNet_preprocessed/<dataset>`.

    Args:
        input_dir: nnU-Net dataset with `imagesTr` and `labelsTr`.
        output_dir: New nnU-Net dataset.
        json_file: Split file. The names can have any form that `standard_name` accepts.
        fold: Fold in the split file.

    Returns:
        The names without an image or a label.

    Raises:
        ValueError: If `output_dir` already holds cases.
    """
    train, val = read_split(json_file, fold)
    for folder in ("imagesTr", "labelsTr"):
        path = os.path.join(output_dir, folder)
        if os.path.isdir(path) and os.listdir(path):
            raise ValueError(f"{path} is not empty. Choose a new dataset folder.")
    missing = set()
    for folder in ("imagesTr", "labelsTr"):
        missing.update(copy_cases(train + val, os.path.join(input_dir, folder), os.path.join(output_dir, folder)))

    cases = _case_index([os.path.join(output_dir, "labelsTr")], ".nii.gz")
    with open(json_file) as f:
        folds = json.load(f)
    splits = []
    for index, names in enumerate(folds):
        split = {}
        for mode in ("train", "val"):
            split[mode] = [cases[key] for key in map(_match_key, names[mode]) if key in cases]
            lost = len(names[mode]) - len(split[mode])
            if lost:
                print(f"Warning: {lost} {mode} cases of fold {index} have no file. splits_final.json omits them.")
        splits.append(split)
    with open(os.path.join(output_dir, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent="\t", separators=(",", ": "))
    return sorted(missing)


def split_sam_data(input_dir: str, output_dir: str, json_file: str, fold: int = 0) -> List[str]:
    """Copy the H5 files of a split into the folders `train_<suffix>` and `val`.

    `<suffix>` is the last part of the split file name, e.g. "n001" for `train_splits_sam_n001.json`.

    Args:
        input_dir: Folder with the H5 files.
        output_dir: Output folder.
        json_file: Split file. The names can have any form that `standard_name` accepts.
        fold: Fold in the split file.

    Returns:
        The names without a file.
    """
    train, val = read_split(json_file, fold)
    suffix = os.path.splitext(os.path.basename(json_file))[0].split("_")[-1]
    missing = copy_cases(train, input_dir, os.path.join(output_dir, f"train_{suffix}"))
    missing += copy_cases(val, input_dir, os.path.join(output_dir, "val"))
    return missing


def split_semantic_data(
    nnunet_dir: str,
    umg_dir: str,
    output_dir: str,
    json_file: str,
    fold: int = 0,
) -> List[str]:
    """Write the training data of the semantic OCT-SAM variant from a split of the nnU-Net dataset.

    The output has one folder `<dataset>_<train|val>` for each dataset. The HCMS and Duke DME B-scans are
    converted from the nnU-Net dataset to H5 files with the label in "masks". The UMG-RP H5 files are copied,
    so their labels stay in "labels/<version>". All files keep their names.
    `semantic_training_split.json` lists the names of the split file for each dataset.

    Args:
        nnunet_dir: nnU-Net dataset with `imagesTr` and `labelsTr`.
        umg_dir: Folder with the UMG-RP H5 files.
        output_dir: Output folder.
        json_file: Split file. The names can have any form that `standard_name` accepts.
        fold: Fold in the split file.

    Returns:
        The names without a file.
    """
    train, val = read_split(json_file, fold)
    split = {dataset: {"train": [], "val": []} for dataset in DATASETS}
    for mode, names in (("train", train), ("val", val)):
        for name in names:
            split[case_source(name)][mode].append(name)

    cases = _case_index([os.path.join(nnunet_dir, "imagesTr")], ".nii.gz")
    missing = []
    for dataset, modes in split.items():
        for mode, names in modes.items():
            out_dir = os.path.join(output_dir, f"{dataset}_{mode}")
            if dataset == "umg_rp":
                missing += copy_cases(names, umg_dir, out_dir)
                continue
            os.makedirs(out_dir, exist_ok=True)
            for name in tqdm(names, desc=f"{dataset} {mode}"):
                case = cases.get(_match_key(name))
                label_path = os.path.join(nnunet_dir, "labelsTr", f"{case}.nii.gz")
                if case is None or not os.path.isfile(label_path):
                    missing.append(name)
                    continue
                image = read_nifti(os.path.join(nnunet_dir, "imagesTr", f"{case}_0000.nii.gz"))
                write_h5_sample(os.path.join(out_dir, f"{case}.h5"), image, read_nifti(label_path))

    with open(os.path.join(output_dir, "semantic_training_split.json"), "w") as f:
        json.dump(split, f, indent="\t", separators=(",", ": "))
    return missing


def create_train_val_splits_for_finetuning(
    out_dir: str,
    input_json: Optional[str] = None,
    names_train: Optional[List[str]] = None,
    names_val: Optional[List[str]] = None,
    sample_sizes_train: Sequence[int] = (100, 50, 25, 10, 5, 1),
    sample_size_val: int = 10,
    output_sam: bool = False,
    random_seed: int = 42,
) -> None:
    """Write nested training subsets with a fixed validation set for the iterative fine-tuning.

    Each subset is drawn from the previous, larger one. The function writes one file
    `train_splits_n<N>.json` (or `train_splits_sam_n<N>.json`) for each sample size.
    With the same seed and input, the result is the same as the files in `doc/train_splits`.

    Args:
        out_dir: Output folder for the split files.
        input_json: Split file with all training ("train") and validation ("val") names.
        names_train: Names for training. They replace the "train" list of `input_json`.
        names_val: Names for validation. They replace the "val" list of `input_json`.
        sample_sizes_train: Sizes of the training subsets, from large to small.
        sample_size_val: Size of the validation set.
        output_sam: Write the names of the UMG-RP H5 files, e.g. "RP012_2023_OD_z08", instead of the input names.
        random_seed: Seed for the random selection.
    """
    if input_json is not None:
        train, val = read_split(input_json)
        names_train = train if names_train is None else names_train
        names_val = val if names_val is None else names_val
    if names_train is None:
        raise ValueError("Pass the names for training, or a split file.")

    def output_names(names):
        return [umg_rp_source_name(name) for name in names] if output_sam else names

    np.random.seed(random_seed)

    val_subset = []
    if names_val is not None:
        selected = np.random.choice(names_val.copy(), size=sample_size_val, replace=False).tolist()
        val_subset = output_names(selected)

    subsets = {}
    current_subset = names_train.copy()
    for n in sample_sizes_train:
        selected = np.random.choice(current_subset, size=n, replace=False).tolist()
        subsets[n] = output_names(selected)
        current_subset = selected

    os.makedirs(out_dir, exist_ok=True)
    prefix = "train_splits_sam" if output_sam else "train_splits"
    for size, subset in subsets.items():
        out_path = os.path.join(out_dir, f"{prefix}_n{str(size).zfill(3)}.json")
        with open(out_path, "w") as f:
            json.dump([{"train": sorted(subset), "val": sorted(val_subset)}], f, indent='\t', separators=(',', ': '))
