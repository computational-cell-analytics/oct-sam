import json
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np
import torch_em
from micro_sam.training import train_sam_for_configuration
from micro_sam.util import export_custom_sam_model


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


def export_model(model_name):
    export_custom_sam_model(
        f"./checkpoints/{model_name}/best.pt", model_type="vit_b", save_path=f"./{model_name}.pt",
        with_segmentation_decoder=True
    )


def train_oct_sam_model(
    model_name: str,
    train_loader,
    val_loader,
    check: bool = False,
):
    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    train_sam_for_configuration(
        name=model_name, train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type="vit_b_medical_imaging",
        verify_n_labels_in_loader=5,
        n_epochs=40,
        strict_decoder_loading=False,
    )


def create_train_val_splits_for_finetuning(
    out_dir: str,
    input_json: Optional[str] = None,
    names_train: Optional[List[str]] = None,
    names_val: Optional[List[str]] = None,
    sample_sizes_train: List[int] = [100, 50, 25, 10, 5, 1],
    sample_size_val: int = 10,
    output_sam: bool = False,
    random_seed: int = 42,
):
    """Create a list of names for training and validation used for iterative finetuning.

    Args:
        out_dir: Output directory for JSON dictionaries.
        input_json: Input JSON featuring all training ('train') and validation ('val') names.
        names_train: List of names for training.
        names_val: List of names for validation.
        sample_sizes_train: Sample sizes for training.
        sample_size_val: Fixed size for validation data.
        output_sam: Output OCT-SAM naming scheme.
        random_seed: Seed for randomization to determine train/val splits.
    """
    if names_train is None and names_val is None and input_json is None:
        raise ValueError("Pass a list of names for training, validation, or a JSON dictionary.")

    if input_json is not None:
        with open(input_json, 'r') as myfile:
            data = myfile.read()
        params = json.loads(data)[0]

        if names_train is None:
            names_train = params["train"]
        if names_val is None:
            names_val = params["val"]

    np.random.seed(random_seed)

    val_subset = []
    if names_val is not None:
        selected = np.random.choice(names_val.copy(), size=sample_size_val, replace=False)
        selected = selected.tolist()
        if output_sam:
            subset_out = []
            for select in selected:
                content = select.split("_")
                name_sam = f"RP{content[1][2:]}_{content[2]}_{content[3]}_z{content[4][1:]}"
                subset_out.append(name_sam)
        else:
            subset_out = selected
        val_subset = subset_out

    subsets = {}
    current_subset = names_train.copy()

    for n in sample_sizes_train:
        # Randomly select n samples from the current subset
        selected = np.random.choice(current_subset, size=n, replace=False)
        selected = selected.tolist()
        if output_sam:
            subset_out = []
            for select in selected:
                content = select.split("_")
                name_sam = f"RP{content[1][2:]}_{content[2]}_{content[3]}_z{content[4][1:]}"
                subset_out.append(name_sam)
        else:
            subset_out = selected

        subsets[n] = subset_out
        current_subset = selected

    # Example of accessing results
    for size, subset in subsets.items():
        new_dic = {}
        if output_sam:
            out_path = os.path.join(out_dir, f"train_splits_sam_n{str(size).zfill(3)}.json")
        else:
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
    Copy files from input_dir to output_dir if the filename contains any of the subset names as a substring.

    Args:
        subset_names: List of names to match (e.g., ['Name_1', 'Name_5'])
        input_dir: Path object to input directory
        output_dir: Path object to output directory

    Returns:
        List of copied files
        List of skipped files
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
