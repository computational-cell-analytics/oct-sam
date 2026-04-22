import argparse
import os
from glob import glob
from pathlib import Path

import h5py
import imageio.v3 as imageio
import napari

# matching of annotation data and output folder name
ANNOTATION_DICT = {
    "annotations_training_cycle_1": "20251126",
    "annotations_training_cycle_2": "20251215",
    "annotations_training_cycle_3": "20260105",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert TIF data from training data cycles into H5 format for network training."
    )
    parser.add_argument("-i", "--input", required=True, help="Input folder containing images and annotations.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output folder.")

    args = parser.parse_args()

    assert os.path.exists(args.input)
    label_files = glob(os.path.join(args.input, "*_committed_objects.tif"))

    for label_path in label_files:
        image_path = label_path.replace("_committed_objects", "")
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        if args.output is not None:
            os.makedirs(args.output, exist_ok=True)
            fname = Path(image_path).stem
            output_path = os.path.join(args.output, f"{fname}.h5")
            with h5py.File(output_path, "a") as f:
                f.create_dataset("image", data=image, compression="gzip")
                f.create_dataset("labels/original", data=labels, compression="gzip")
        else:
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(labels)
            napari.run()


if __name__ == "__main__":
    main()
