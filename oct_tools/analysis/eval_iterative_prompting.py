import os

import h5py
import imageio.v3 as imageio
import numpy as np

import pandas as pd


def create_individual_tif_data(
    input_dir: str,
    image_dir: str,
    label_dir: str,
    image_key: str = "image",
    label_key: str = "labels/edit_v3",
):
    """Create individual TIF data for image and label from H5 data.

    Args:
        input_dir: Input directory with H5 data.
        image_dir: Output directory for images in TIF format.
        label_dir: Output directory for labels in TIF format.
        image_key: Input key in H5 file for image data.
        label_key: Input key in H5 file for label data.
    """
    h5_paths = [entry.path for entry in os.scandir(input_dir) if ".h5" in entry.name]
    h5_paths.sort()
    for h5_path in h5_paths:
        image = np.array(h5py.File(h5_path, "r")[image_key])
        label = np.array(h5py.File(h5_path, "r")[label_key])
        base_name = os.path.basename(h5_path).split(".")[0]

        unique_ids = np.unique(label)[1:]
        for unique_id in unique_ids:
            unique_label_dir = os.path.join(label_dir, str(unique_id))
            os.makedirs(unique_label_dir, exist_ok=True)

            unique_image_dir = os.path.join(image_dir, str(unique_id))
            os.makedirs(unique_image_dir, exist_ok=True)

            out_path_label = os.path.join(unique_label_dir, f"{base_name}.tif")
            label_tmp = (label == unique_id)
            imageio.imwrite(out_path_label, label_tmp)

            out_path_image = os.path.join(unique_image_dir, f"{base_name}.tif")
            imageio.imwrite(out_path_image, image)


def read_results_single(output_folder, iterations=[0, 1, 3]):
    result_dict = {}
    results_dir = os.path.join(output_folder, "results", "iterative_prompting_without_mask")
    csv_files = [entry.path for entry in os.scandir(results_dir) if ".csv" in entry.name]
    csv_files.sort()
    for file_path in csv_files:
        dict_type = os.path.basename(file_path).split("_")[-1].split(".")[0]
        df = pd.read_csv(file_path)
        # df.rename(columns={"Unnamed: 0": "Iteration"}, inplace=True)
        acc_dict = {}
        for i in iterations:
            acc_dict[i] = {}
            acc_dict[i]["precision"] = float(df.at[i, 'Precision'])
            acc_dict[i]["recall"] = float(df.at[i, 'Recall'])
            acc_dict[i]["f1-score"] = float(df.at[i, 'F1 Score'])
        result_dict[dict_type] = acc_dict
    return result_dict


def eval_iter_prompts_network_single(dataset_dir, iterations=[0, 1, 3]):
    network_dict = {}
    layer_indexes = [int(entry.name) for entry in os.scandir(dataset_dir)]
    layer_results = [entry.path for entry in os.scandir(dataset_dir)]
    layer_indexes.sort()
    layer_results.sort()
    for (layer_index, layer_result) in zip(layer_indexes, layer_results):
        network_dict[layer_index] = read_results_single(layer_result, iterations=iterations)
    return network_dict


def eval_iter_prompts_networks_multi(main_dir, networks, iterations=[0, 1, 2, 3]):
    total_dict = {}
    for network in networks:
        dataset_dir = os.path.join(main_dir, network)
        network_dict = eval_iter_prompts_network_single(dataset_dir, iterations)
        total_dict[network] = network_dict
    return total_dict
