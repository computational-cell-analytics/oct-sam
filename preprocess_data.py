import argparse
import os

import numpy as np
import pandas as pd
from typing import Optional

import imageio.v3 as imageio


def get_dict(table_path: str, data_dir: str):
    """Dictionary for OCT data from formatted excel table.
    """
    table = pd.read_excel(table_path)
    layer_columns = [col for col in table.columns if col != "Image"]
    data_dict = []

    for i, row in table.iterrows():
        file_name = row.Image
        folder_name = file_name.split("_")[0]
        dic = {"file_name": file_name}
        dic["folder_name"] = folder_name
        seg_classes = [layer.split(" ")[1].split("-")[0] for layer in layer_columns]
        dic["seg_classes"] = seg_classes
        dic["label_ref"] = [row[col] for col in layer_columns]
        dic["img_path"] = os.path.join(data_dir, folder_name, file_name + "_croppedb.tif")
        dic["seg_path"] = os.path.join(data_dir, folder_name, file_name + "_maskedb.tif")
        data_dict.append(dic)

    return data_dict


def get_arrays(oct_dict: dict, cut_legend: Optional[int] = None):
    """Read formatted image and segmentation arrays from data.
    The data is cropped to omit a legend visible in the image.
    """
    img_arr = imageio.imread(oct_dict["img_path"])
    seg_arr = imageio.imread(oct_dict["seg_path"])
    if cut_legend is not None:
        img_arr = img_arr[:-cut_legend, :]
        seg_arr = seg_arr[:-cut_legend, :]
    labels = [num + 1 for num, i in enumerate(oct_dict["label_ref"]) if i == 1]
    unique, counts = np.unique(seg_arr, return_counts=True)
    for label, u in zip(labels, list(unique[1:])):
        seg_arr[seg_arr == u] = label
    return img_arr, seg_arr


def preprocess_data(table_path: str, data_dir: str, out_dir: str):
    """Preprocess OCT data with information given from an excel table.
    """
    data_dict = get_dict(table_path, data_dir)
    for num, dic in enumerate(data_dict):
        file_name = dic["file_name"]
        img_arr, seg_arr = get_arrays(dic, cut_legend=65)
        img_out = os.path.join(out_dir, f"{file_name}.tif")
        imageio.imwrite(img_out, img_arr)
        seg_out = os.path.join(out_dir, f"{file_name}_annotations.tif")
        imageio.imwrite(seg_out, seg_arr)


def main():

    parser = argparse.ArgumentParser(
        description="Script to prerocess OCT data for training with micro-sam.")

    parser.add_argument('-t', '--table', type=str, required=True, help="Input file in n5 / ome-zarr format.")
    parser.add_argument('-d', "--data_dir", type=str,
                        default="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/20250401",
                        help="Data directory.")
    parser.add_argument('-o', "--output_dir", type=str,
                        default="/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data",
                        help="Output directory for training data.")

    args = parser.parse_args()

    preprocess_data(args.table, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
