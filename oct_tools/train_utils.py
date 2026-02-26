import json
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import torch_em
import vigra
from skimage.measure import regionprops


class BoundaryAndDistanceTransform:
    eps = 1e-7

    def __init__(self):
        self.label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True, ndim=2)

    def compute_normalized_directed_distances(self, mask, boundaries, bb, distances):
        cropped_mask = mask[bb]
        inv_mask = ~cropped_mask

        cropped_boundary_mask = boundaries[bb]
        this_distances = np.abs(vigra.filters.vectorDistanceTransform(cropped_boundary_mask))
        this_distances[inv_mask] = 0

        spatial_axes = tuple(range(mask.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        distances[bb][cropped_mask] = this_distances[cropped_mask]
        return distances

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        fg_and_bd = self.label_transform(labels)
        assert fg_and_bd.shape[0] == 2
        boundaries = fg_and_bd[1].astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        labels = labels + 1
        props = regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim)) for prop in props
        }

        # Compute how many distance channels we have.
        n_channels = 2

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), 1, dtype="float32")
        for prop in props:
            label_id = prop.label
            mask = labels == label_id
            distances = self.compute_normalized_directed_distances(
                mask, boundaries, bounding_boxes[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        return np.concatenate([fg_and_bd, distances], axis=0)


def create_train_val_splits(
    out_dir: str,
    input_json: Optional[str] = None,
    names_train: Optional[List[str]] = None,
    names_val: Optional[List[str]] = None,
    sample_sizes_train: List[int] = [100, 50, 25, 10, 5, 1],
    sample_size_val: int = 10,
):
    """Create a list of names for training and validation.

    Args:
        out_dir: Output directory for JSON dictionaries.
        input_json: Input JSON featuring all training ('train') and validation ('val') names.
        names_train: List of names for training.
        names_val: List of names for validation.
        sample_sizes_train: Sample sizes for training.
        sample_size_val: Fixed size for validation data.
    """
    if names_train is None and names_val is None and input_json is None:
        raise ValueError("Pass a list of names for training, validation, or a JSON dictionary.")

    if input_json is not None:
        with open(input_json, 'r') as myfile:
            data = myfile.read()
        params = json.loads(data)

        if names_train is None:
            names_train = params["train"]
        if names_val is None:
            names_val = params["val"]

    np.random.seed(42)

    val_subset = []
    if names_val is not None:
        selected = np.random.choice(names_val.copy(), size=sample_size_val, replace=False)
        val_subset = selected.tolist()

    subsets = {}
    current_subset = names_train.copy()

    for n in sample_sizes_train:
        # Randomly select n samples from the current subset
        selected = np.random.choice(current_subset, size=n, replace=False)

        subsets[n] = selected.tolist()
        current_subset = selected

    # Example of accessing results
    for size, subset in subsets.items():
        new_dic = {}
        out_path = os.path.join(out_dir, f"train_splits_n{str(size).zfill(3)}.json")
        new_dic["train"] = sorted(subset)
        new_dic["val"] = sorted(val_subset)
        dic_list = [new_dic]
        with open(out_path, "w") as f:
            json.dump(dic_list, f, indent='\t', separators=(',', ': '))


def copy_files_by_subset(
    subset_names: List[str],
    input_dir: str,
    output_dir: str,
) -> Tuple[List[str], List[str]]:
    """
    Copy files from input_dir to output_dir if the filename contains any of the subset names as substring.

    Args:
        subset_names: List of names to match (e.g., ['Name_1', 'Name_5'])
        input_dir: Path object to input directory
        output_dir: Path object to output directory
    """
    copied_files = []
    skipped_files = []

    # Check if input directory exists
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Iterate through all files in input directory
    for ff in os.scandir(input_dir):
        file_path = ff.path
        filename = ff.name
        if os.path.isfile(ff.path):

            # Check if any name in subset is a substring of the filename
            for name in subset_names:
                if name in filename:
                    output_path = os.path.join(output_dir, filename)

                    try:
                        shutil.copy2(file_path, output_path)
                        copied_files.append(filename)
                        break
                    except Exception as e:
                        print(f"Error copying {filename}: {e}")
                        skipped_files.append(filename)
                        break

    return copied_files, skipped_files
