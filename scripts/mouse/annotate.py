import argparse
import os
import tempfile

import imageio.v3 as imageio
from micro_sam.sam_annotator import image_series_annotator


def collect_bscans(image_dir, label_dir, slice_dir):
    """Pair every B-scan in image_dir with its initial label in label_dir.

    micro_sam names a saved label after the image file, so each B-scan of a stack is written to
    slice_dir as `<name>_z<slice>.tif`. This matches the output names of `oct_tools.apply_nnunet`.
    """
    images, labels, missing = [], [], []
    for fname in sorted(os.listdir(image_dir)):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in (".tif", ".tiff"):
            continue
        image_path = os.path.join(image_dir, fname)
        data = imageio.imread(image_path)
        label_path = os.path.join(label_dir, f"{name}.tif")

        # A last axis of 3 is one RGB B-scan, as in oct_tools.apply_nnunet._convert_to_nnunet_format.
        if data.ndim != 3 or data.shape[-1] == 3:
            images.append(image_path)
            labels.append(label_path)
            if not os.path.exists(label_path):
                missing.append(label_path)
            continue

        label_stack = imageio.imread(label_path) if os.path.exists(label_path) else None
        for z, bscan in enumerate(data):
            slice_path = os.path.join(slice_dir, f"{name}_z{z:03}.tif")
            imageio.imwrite(slice_path, bscan)
            images.append(slice_path)
            slice_label_path = os.path.join(label_dir, f"{name}_z{z:03}.tif")
            if os.path.exists(slice_label_path):
                labels.append(slice_label_path)
            elif label_stack is not None:
                labels.append(label_stack[z])
            else:
                labels.append(slice_label_path)
                missing.append(slice_label_path)

    if missing:
        raise FileNotFoundError("Missing labels:\n" + "\n".join(missing))
    return images, labels


def main():
    parser = argparse.ArgumentParser(
        description="Correct the retina segmentation of mouse B-scans in napari with micro_sam."
    )
    parser.add_argument("-i", "--image_dir", type=str, required=True,
                        help="Input directory with B-scans in TIF format, 2D or stacks.")
    parser.add_argument("-l", "--label_dir", type=str, required=True,
                        help="Directory with the segmentations of oct_tools.apply_nnunet.")
    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for the corrected labels.")
    args = parser.parse_args()

    # The annotator reads the stack slices until napari closes.
    # The decoder of vit_b_medical_imaging does not load into the micro_sam 1.7 instance decoder.
    with tempfile.TemporaryDirectory() as slice_dir:
        images, labels = collect_bscans(args.image_dir, args.label_dir, slice_dir)
        image_series_annotator(
            images, args.output_dir, model_type="vit_b_medical_imaging",
            initial_segmentations=labels, skip_segmented=True, prefer_decoder=False,
        )


if __name__ == "__main__":
    main()
