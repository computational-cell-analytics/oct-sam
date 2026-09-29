import argparse

from oct_tools.apply_nnunet import _convert_nifti_to_tif


def main():
    parser = argparse.ArgumentParser(
        description="Convert NIfTI inference data to TIF format."
    )

    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input directory containing NIfTI files.")
    parser.add_argument("-o", "--output", type=str,
                        help="Output directory for TIF files. Default: Same as input directory.")
    parser.add_argument("--label", action="store_true",
                        help="Create label data.")

    args = parser.parse_args()

    _convert_nifti_to_tif(
        input_folder=args.input,
        output_folder=args.output or args.input,
        label_data=args.label,
    )


if __name__ == "__main__":
    main()
