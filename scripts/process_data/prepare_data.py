import argparse

from oct_tools.data_processing import DATASETS, SPACING, prepare_dataset, read_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert a training dataset for nnU-Net (NIfTI files in imagesTr and labelsTr) "
        "or for OCT-SAM (one H5 file for each B-scan with the datasets 'image' and 'masks')."
    )
    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Root folder of the dataset. For UMG-RP, the folder with the H5 files.")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output folder.")
    parser.add_argument("--dataset", type=str, required=True, choices=DATASETS, help="Dataset of the input.")
    parser.add_argument("--format", type=str, default="nnunet", choices=["nnunet", "h5"],
                        help="Output format: 'nnunet' for nnU-Net, 'h5' for OCT-SAM. Default: nnunet.")
    parser.add_argument("-l", "--label_key", type=str, default="edit_v3",
                        help="Label version in the UMG-RP H5 files. Default: 'edit_v3'.")
    parser.add_argument("--binary", action="store_true",
                        help="Write a binary label with 0 for the background and 1 for the retinal layers.")
    parser.add_argument("--keep_is_os", action="store_true",
                        help="Keep the 8 HCMS layers. By default, the inner and outer photoreceptor segments "
                        "become one layer, the ellipsoid zone.")
    parser.add_argument("--legacy_names", action="store_true",
                        help="Use the names of the earlier conversions instead of the standard names, e.g. "
                        "oct_20001010 for nnU-Net and Subject_01_z000.h5 for H5 instead of duke_01_z10. "
                        "See doc/training_data.md.")

    args = parser.parse_args()
    if args.keep_is_os and args.dataset != "hcms":
        parser.error("--keep_is_os applies only to --dataset hcms.")

    samples = read_dataset(args.dataset, args.input_dir, label_key=args.label_key, merge_is_os=not args.keep_is_os)
    count = prepare_dataset(
        samples, args.output_dir, args.format, SPACING[args.dataset], binary=args.binary, legacy_names=args.legacy_names
    )
    if count == 0:
        raise ValueError(f"{args.input_dir} has no B-scans of {args.dataset}. Check -i and, for UMG-RP, -l.")
    print(f"Wrote {count} B-scans to {args.output_dir}.")


if __name__ == "__main__":
    main()
