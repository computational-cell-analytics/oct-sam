import argparse

from oct_tools.apply_nnunet import _convert_to_nnunet_format


def main():
    parser = argparse.ArgumentParser(
        description="Convert input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True,
                        help="Input directory containing image data.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for converted NIfTI files.")
    parser.add_argument("-f", "--file_format", type=str, default=None, choices=[None, "h5", "tif"],
                        help="File format of input data. Default: Most frequent file type in input directory.")

    args = parser.parse_args()

    _convert_to_nnunet_format(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        file_format=args.file_format,
    )


if __name__ == "__main__":
    main()
