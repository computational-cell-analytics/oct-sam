import argparse
import json
import os

from oct_tools.train_utils import copy_files_by_subset


def create_subset_sam_training_data(
    input_dir,
    output_dir,
    json_file,
):
    with open(json_file, 'r') as myfile:
        data = myfile.read()
    params = json.loads(data)[0]
    train_names = params["train"]
    val_names = params["val"]

    train_suffix = os.path.basename(json_file).split("_")[-1].split(".")[0]

    # copy training data
    train_in = os.path.join(input_dir)
    if train_suffix is None:
        train_out = os.path.join(output_dir)
    else:
        train_out = os.path.join(output_dir, f"train_{train_suffix}")

    os.makedirs(train_out, exist_ok=True)
    copied, skipped = copy_files_by_subset(train_names, train_in, train_out)
    if copied:
        print(f"Images copied: {copied[:5]}{'...' if len(copied) > 5 else ''}")
    if skipped:
        print(f"Images not copied: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # copy validation data
    train_in = os.path.join(input_dir)
    train_out = os.path.join(output_dir, "val")

    os.makedirs(train_out, exist_ok=True)
    copied, skipped = copy_files_by_subset(val_names, train_in, train_out)
    if copied:
        print(f"Labels copied: {copied[:5]}{'...' if len(copied) > 5 else ''}")
    if skipped:
        print(f"Labels not copied: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def create_subset_nnunet_training_data(
    input_dir: str,
    output_dir: str,
    json_file: str,
):
    """Create subset of nnU-Net training data based on a 'train'/'val' split in a JSON dictionary.
    The function scans the input directory, which should be in the $nnUNet_raw directory for the
    subfolders 'imagesTr' and 'labelsTr'.
    All files which contain the names from the JSON dictionary are copied to the output directory.

    Args:
        input_dir: Input directory for dataset in $nnUNet_raw directory.
        output_dir: Output directory for subset of data in $nnUNet_raw directory.
        json_file: JSON dictionary with names for training and validation.
    """
    # read names
    with open(json_file, 'r') as myfile:
        data = myfile.read()
    params = json.loads(data)[0]
    names = params["train"]
    names += params["val"]

    # copy images
    train_in = os.path.join(input_dir, "imagesTr")
    train_out = os.path.join(output_dir, "imagesTr")
    os.makedirs(train_out, exist_ok=True)
    copied, skipped = copy_files_by_subset(names, train_in, train_out)
    if copied:
        print(f"Images copied: {copied[:5]}{'...' if len(copied) > 5 else ''}")
    if skipped:
        print(f"Images not copied: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")

    # copy labels
    train_in = os.path.join(input_dir, "labelsTr")
    train_out = os.path.join(output_dir, "labelsTr")
    os.makedirs(train_out, exist_ok=True)
    copied, skipped = copy_files_by_subset(names, train_in, train_out)
    if copied:
        print(f"Labels copied: {copied[:5]}{'...' if len(copied) > 5 else ''}")
    if skipped:
        print(f"Labels not copied: {skipped[:5]}{'...' if len(skipped) > 5 else ''}")


def main():
    parser = argparse.ArgumentParser(
        description="Create training data for nnU-Net based on JSON dictionary."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-j", "--json", type=str, required=True)
    parser.add_argument("--sam", action="store_true",
                        help="Use splits for SAM pre-training dataset.")

    args = parser.parse_args()

    if args.sam:
        create_subset_sam_training_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            json_file=args.json,
        )

    else:
        create_subset_nnunet_training_data(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            json_file=args.json,
        )


if __name__ == "__main__":
    main()
