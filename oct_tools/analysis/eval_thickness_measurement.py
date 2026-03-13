import json
import os
from typing import List, Optional

import nibabel as nib
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from oct_tools.segmentation_utils import _thickness_at_reference

LAYER_MAPPING = {
   1: "RNFL",
   2: "GCIPL",
   3: "INL",
   4: "OPL",
   5: "ONL",
   6: "EZ",
   7: "RPE"
}


def eval_excel(
    excel_in: str,
    json_out: Optional[str] = None,
) -> Optional[dict]:
    df = pd.read_excel(excel_in)
    df = df.reset_index()  # Make sure indexes pair with number of rows
    val_dict = {}
    slices = []
    current_dict = {}
    for _, row in df.iterrows():
        # identify new patient dataset
        if isinstance(row["Patient"], str):
            # skip if first iteration
            if len(slices) != 0:
                slices.sort()
                current_dict["slices"] = slices
                current_dict = {key: current_dict[key] for key in sorted(current_dict.keys())}
                val_dict[current_dict["patient_id"]] = current_dict

            current_dict = {}
            slices = []
            p_split = row["Patient"].strip().split(" ")
            patient_id = f"RP{p_split[1][2:-1].zfill(3)}"
            meas_year = p_split[3][:-1]
            eye_id = p_split[-1]
            current_dict["patient_id"] = patient_id
            current_dict["meas_year"] = meas_year
            current_dict["eye_id"] = eye_id
        # new slice
        if isinstance(row["Slice"], str):
            slice_index = row["Slice"].strip().split("z")[1]
            slices.append(int(slice_index))
            current_dict[f"z{slice_index.zfill(2)}"] = {}
        # new vertical x position
        if isinstance(row["Verticalx"], str):
            x_pos = row["Verticalx"].strip().split("x")[1]
            current_dict[f"z{slice_index.zfill(2)}"][f"x{x_pos.zfill(4)}"] = {}
        # read out layer thicknesses
        for col in ["CFT", "RNFL", "GCIPL", "INL", "OPL", "ONL", "EZ", "RPE"]:
            if isinstance(row[col], float) and not np.isnan(row[col]):
                current_dict[f"z{slice_index.zfill(2)}"][f"x{x_pos.zfill(4)}"][col] = row[col]

    val_dict = {key: val_dict[key] for key in sorted(val_dict.keys())}

    if json_out is None:
        return val_dict
    else:
        with open(json_out, "w") as f:
            json.dump(val_dict, f, indent='\t', separators=(',', ': '))


def get_thickness_dict_for_x(arr, x_pos, spacing, layer_mapping=LAYER_MAPPING):
    thickness_dict = {}
    props = regionprops(arr, spacing=spacing)
    for prop in props:
        mask_all = (arr == prop.label)
        if prop.label not in layer_mapping.keys():
            raise ValueError(f"Unknown segmentation label {prop.label}.")

        thickness = _thickness_at_reference(mask_all, x_pos, spacing)
        thickness_dict[layer_mapping[prop.label]] = thickness
    return thickness_dict


def init_eval_dict(layers):
    eval_dict = {}
    for layer in layers:
        eval_dict[layer] = {
            "abs": [],
            "rel": [],
            "tp": 0,
            "fn": 0,
            "fp": 0,
        }
    return eval_dict


def eval_thickness_nnunet_single(
    slice_dict: dict,
    file_path: str,
    voxel_size: tuple = None,
    exclude_missing: bool = True,
    eval_dict: Optional[dict] = None,
    layers: List[str] = [items for _, items in LAYER_MAPPING.items()],
):
    nib_data = nib.load(file_path)
    arr = nib_data.get_fdata()
    arr = np.array(arr).astype("int64")
    if voxel_size is None:
        data_hdr = nib_data.header
        voxel_size = (data_hdr["srow_x"][0], data_hdr["srow_y"][1])

    # initialize new evaluation dictionary
    if eval_dict is None:
        eval_dict = init_eval_dict(layers)

    for key, items in slice_dict.items():
        x_pos = int(key[1:])
        thickness_dict = get_thickness_dict_for_x(arr, x_pos=x_pos, spacing=voxel_size)
        # exclude central fovea thickness (CFT)
        ref_layers = [ref for ref in list(items.keys()) if ref in layers]
        nnunet_layers = list(thickness_dict.keys())

        for nnunet_layer in nnunet_layers:
            if nnunet_layer not in ref_layers:
                eval_dict[nnunet_layer]["fp"] += 1

        for ref in ref_layers:
            if ref not in nnunet_layers:
                eval_dict[ref]["fn"] += 1
                if exclude_missing:
                    continue
                else:
                    eval_dict[ref]["abs"].append(items[ref])
                    eval_dict[ref]["rel"].append(1)
            else:
                eval_dict[ref]["tp"] += 1
                eval_dict[ref]["abs"].append(abs(items[ref] - thickness_dict[ref]))
                eval_dict[ref]["rel"].append(abs(items[ref] - thickness_dict[ref]) / items[ref])
    return eval_dict


def eval_thickness_nnunet_multi(
    measurement_json: str,
    nnunet_inference_dir: str,
    out_path: str,
    layers: List[str] = [items for _, items in LAYER_MAPPING.items()],
    round_decimal: Optional[int] = 3,
):
    """Evaluate nnU-Net segmentation by calculating the error in comparison to manual determined thicknesses.

    Args:
        measurement_json: JSON dictionary featuring layer thicknesses for x-positions of different datasets.
        nnunet_inference_dir: Directory containing the segmentation of single test images.
        out_path: Output path for JSON dictionary with error values.
        layers: List of layers.
    """
    with open(measurement_json, 'r') as myfile:
        data = myfile.read()
    params = json.loads(data)

    eval_dict = init_eval_dict(layers)

    for key, items in params.items():
        patient_id = items["patient_id"]
        meas_year = items["meas_year"]
        eye_id = items["eye_id"]
        slices = items["slices"]
        for slice_index in slices:
            file_name = f"oct_{patient_id}_{meas_year}_{eye_id}_z{str(slice_index).zfill(2)}.nii.gz"
            file_path = os.path.join(nnunet_inference_dir, file_name)
            slice_str = f"z{str(slice_index).zfill(2)}"
            eval_dict = eval_thickness_nnunet_single(
                slice_dict=items[slice_str], file_path=file_path, eval_dict=eval_dict)

    error_dict = {}
    for key, items in eval_dict.items():
        layer_dict = {}
        layer_dict["mean_abs_error"] = np.mean(items["abs"])
        layer_dict["std_abs_error"] = np.std(items["abs"])
        layer_dict["mean_rel_error"] = np.mean(items["rel"]) * 100
        layer_dict["std_rel_error"] = np.std(items["rel"]) * 100
        layer_dict["tp"] = items["tp"]
        layer_dict["fn"] = items["fn"]
        layer_dict["fp"] = items["fp"]
        layer_dict["precision"] = items["tp"] / (items["tp"] + items["fp"])
        layer_dict["recall"] = items["tp"] / (items["tp"] + items["fn"])
        layer_dict["f1-score"] = (layer_dict["precision"] + layer_dict["recall"]) / 2

        if round_decimal is not None:
            for kkey, iitems in layer_dict.items():
                if isinstance(iitems, float):
                    layer_dict[kkey] = round(iitems, round_decimal)

        error_dict[key] = layer_dict

    with open(out_path, "w") as f:
        json.dump(error_dict, f, indent='\t', separators=(',', ': '))
