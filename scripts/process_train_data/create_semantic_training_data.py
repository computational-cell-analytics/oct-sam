import argparse
import json
import os

import h5py
import nibabel as nib
import numpy as np

from oct_tools.train_utils import copy_files_by_subset


def create_semantic_sam_training_data(
    output_dir: str,
    nnunet_dir: str,
    json_file: str,
    dorothea_data_dir: str,
):
    """Create subset of octSAM training data based on a 'train'/'val' split in a JSON dictionary.
    The function scans the input directory for both training and validation data.
    All files which contain the names from the JSON dictionary are copied to the output directory.

    Args:
        input_dir: Input directory for dataset.
        output_dir: Output directory for subset of data.
        json_file: JSON dictionary with names for training and validation.
        dorothea_dir: Directory containing data from local annotations.
    """
    with open(json_file, 'r') as myfile:
        data = myfile.read()
    params = json.loads(data)[0]
    train_names = params["train"]
    val_names = params["val"]

    data_dict = {
        "hcms": {"train": [], "val": []},
        "duke_dme": {"train": [], "val": []},
        "dorothea": {"train": [], "val": []},
    }
    for (names, data_mode) in zip([train_names, val_names], ["train", "val"]):
        for name in names:
            contents = name.split("_")
            nnunet_data_prefix = contents[1][:2]
            if nnunet_data_prefix in ["11", "12"]:
                data_dict["hcms"][data_mode].append(name)

            if nnunet_data_prefix in ["20"]:
                data_dict["duke_dme"][data_mode].append(name)

            if nnunet_data_prefix in ["30"]:
                middle_part = name[6:]
                middle_part = middle_part[:-3] + "z" + middle_part[-2:]
                input_name = f"RP{middle_part}"
                data_dict["dorothea"][data_mode].append(input_name)

        for key, item in data_dict.items():
            print(f"Copying {data_mode} files for {key}.")
            if key == "hcms":
                train_out = os.path.join(output_dir, f"hcms_{data_mode}")
                os.makedirs(train_out, exist_ok=True)
                for image_name in item[data_mode]:
                    out_path = os.path.join(train_out, f"{image_name}.h5")
                    if os.path.isfile(out_path):
                        continue

                    # image
                    nib_data = nib.load(os.path.join(nnunet_dir, "imagesTr", f"{image_name}_0000.nii.gz"))
                    img = nib_data.get_fdata()
                    img = np.array(img)

                    # label
                    nib_data = nib.load(os.path.join(nnunet_dir, "labelsTr", f"{image_name}.nii.gz"))
                    seg = nib_data.get_fdata()
                    seg = np.array(seg).astype("int64")
                    with h5py.File(out_path, "a") as f:
                        f.create_dataset("image", data=img, compression="gzip")
                        f.create_dataset("masks", data=seg, compression="gzip")

            elif key == "duke_dme":
                train_out = os.path.join(output_dir, f"duke_dme_{data_mode}")
                os.makedirs(train_out, exist_ok=True)

                for image_name in item[data_mode]:
                    out_path = os.path.join(train_out, f"{image_name}.h5")
                    if os.path.isfile(out_path):
                        continue

                    # image
                    nib_data = nib.load(os.path.join(nnunet_dir, "imagesTr", f"{image_name}_0000.nii.gz"))
                    img = nib_data.get_fdata()
                    img = np.array(img)

                    # label
                    nib_data = nib.load(os.path.join(nnunet_dir, "labelsTr", f"{image_name}.nii.gz"))
                    seg = nib_data.get_fdata()
                    seg = np.array(seg).astype("int64")
                    with h5py.File(out_path, "a") as f:
                        f.create_dataset("image", data=img, compression="gzip")
                        f.create_dataset("masks", data=seg, compression="gzip")

            elif key == "dorothea":
                train_out = os.path.join(output_dir, f"dorothea_{data_mode}")
                os.makedirs(train_out, exist_ok=True)
                copied, skipped = copy_files_by_subset(item[data_mode], dorothea_data_dir, train_out)

        out_path = os.path.join(output_dir, "semantic_training_split.json")
        with open(out_path, "w") as f:
            json.dump(data_dict, f, indent='\t', separators=(',', ': '))


def main():
    parser = argparse.ArgumentParser(
        description="Create training data for semantic segmentation using Medico-SAM based on nnU-Net training data."
    )

    parser.add_argument("-n", "--nnunet_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-j", "--json", type=str, required=True)
    parser.add_argument("--dorothea_dir", type=str, required=True)

    args = parser.parse_args()
    create_semantic_sam_training_data(
            nnunet_dir=args.nnunet_dir,
            output_dir=args.output_dir,
            json_file=args.json,
            dorothea_data_dir=args.dorothea_dir,
        )


if __name__ == "__main__":
    main()
