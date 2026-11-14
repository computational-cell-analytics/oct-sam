import argparse

from oct_tools.data_processing import split_nnunet_dataset, split_sam_data, split_semantic_data


def main():
    parser = argparse.ArgumentParser(
        description="Copy the B-scans of a train/val split from doc/train_splits into a new training dataset."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="nnU-Net dataset with imagesTr and labelsTr, or for '--layout sam' a folder of H5 files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("-j", "--json", type=str, required=True,
                        help="Split file. The names can be standard or legacy names. See doc/training_data.md.")
    parser.add_argument("--fold", type=int, default=0, help="Fold in the split file. Default: 0.")
    parser.add_argument(
        "--layout", type=str, default="nnunet", choices=["nnunet", "sam", "semantic"],
        help="'nnunet': a new nnU-Net dataset with all cases of the split, and splits_final.json. "
        "'sam': the folders train_<suffix> and val, e.g. train_n001 for train_splits_sam_n001.json. "
        "'semantic': the folders <dataset>_train and <dataset>_val for train_oct_sam_semantic.py. Default: nnunet.",
    )
    parser.add_argument("--umg_dir", type=str, default=None,
                        help="Folder with the UMG-RP H5 files. Only for '--layout semantic'.")

    args = parser.parse_args()

    if args.layout == "nnunet":
        missing = split_nnunet_dataset(args.input_dir, args.output_dir, args.json, args.fold)
    elif args.layout == "sam":
        missing = split_sam_data(args.input_dir, args.output_dir, args.json, args.fold)
    else:
        if args.umg_dir is None:
            parser.error("'--layout semantic' needs --umg_dir.")
        missing = split_semantic_data(args.input_dir, args.umg_dir, args.output_dir, args.json, args.fold)

    if missing:
        print(f"Warning: {len(missing)} cases of the split have no file: {missing}")


if __name__ == "__main__":
    main()
