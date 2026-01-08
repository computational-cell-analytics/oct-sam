import os
from glob import glob
from pathlib import Path

import h5py
import imageio.v3 as imageio
import napari

folder = "../../data/annotations_training_cycle_3"
export_folder = "../../data/training_data/20260105"


def main():
    assert os.path.exists(folder), folder
    label_files = glob(os.path.join(folder, "*_committed_objects.tif"))
    export = True
    for label_path in label_files:
        image_path = label_path.replace("_committed_objects", "")
        fname = Path(image_path).stem
        if fname == "RP41_210458_z40":
            image_path = os.path.join(folder, "RP41_21045_z40.tif")
        assert os.path.exists(image_path), fname
        image = imageio.imread(image_path)
        labels = imageio.imread(label_path)

        if export:
            os.makedirs(export_folder, exist_ok=True)
            output_path = os.path.join(export_folder, f"{fname}.h5")
            if os.path.exists(output_path):
                continue
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
