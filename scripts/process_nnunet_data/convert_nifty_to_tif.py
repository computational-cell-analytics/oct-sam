import argparse
import os

import imageio.v3 as imageio
import nibabel as nib


def convert_nifty_to_tif(input_folder, output_folder):
    file_paths = [entry.path for entry in os.scandir(input_folder) if ".nii.gz" in entry.name]
    file_paths.sort()

    for ff in file_paths:
        base_name = os.path.basename(ff)
        base_name = base_name.split(".nii.gz")[0]
        out_path = os.path.join(output_folder, f"{base_name}.tif")
        my_img = nib.load(ff)
        arr = my_img.get_fdata()
        imageio.imwrite(out_path, arr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5 input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)

    args = parser.parse_args()

    convert_nifty_to_tif(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
    )


if __name__ == "__main__":
    main()
