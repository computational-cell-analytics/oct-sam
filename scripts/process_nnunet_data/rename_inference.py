import argparse
import os

import imageio.v3 as imageio
import nibabel as nib
import numpy as np


def rename_nnunet_inference(pred_folder, ref_folder, output_folder=None, dataset_id="01"):
    if output_folder is None:
        output_folder = pred_folder

    ref_files = [entry.path for entry in os.scandir(ref_folder) if ".h5" in entry.name]
    ref_files.sort()

    pred_files = [entry.path for entry in os.scandir(pred_folder) if ".nii.gz" in entry.name]
    pred_files.sort()

    os.makedirs(output_folder, exist_ok=True)

    for ff in ref_files:

        base_name = os.path.basename(ff).split(".h5")[0]
        scan_id = base_name.split("_")[0][2:]

        slice_index = base_name[-2:]

        nnunet_identifier = f"{dataset_id}{str(scan_id).zfill(3)}{str(slice_index).zfill(3)}"
        for f in pred_files:
            if nnunet_identifier in f:
                out_path = os.path.join(output_folder, f"{base_name}.tif")
                my_img = nib.load(f)
                arr = my_img.get_fdata()
                arr = np.array(arr).astype("int64")
                imageio.imwrite(out_path, arr)


def main():
    parser = argparse.ArgumentParser(
        description="Convert h5 input data into the nnU-Net format for inference."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-r", "--ref_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, default=None)

    args = parser.parse_args()

    rename_nnunet_inference(
        pred_folder=args.input_dir,
        ref_folder=args.ref_dir,
        output_folder=args.output_dir,
    )


if __name__ == "__main__":
    main()
