import os

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

ROOT = "./data"

table = pd.read_excel("segmented_images_reformatted.xlsx")

view = False

layer_columns = [col for col in table.columns if col != "Image"]

for i, row in table.iterrows():
    file_name = row.Image
    folder_name = file_name.split("_")[0]

    image_path = os.path.join(ROOT, folder_name, f"{file_name}_croppedb.tif")
    assert os.path.exists(image_path), image_path
    label_path = os.path.join(ROOT, folder_name, f"{file_name}_maskedb.tif")
    assert os.path.exists(label_path), label_path

    image, labels = imageio.imread(image_path), imageio.imread(label_path)
    n_labels = len(np.unique(labels)) - 1

    n_expected_labels = sum([row[name] for name in layer_columns])
    if n_labels != n_expected_labels:
        print(file_name)
        print(n_expected_labels)
        print(n_labels)

    if view:
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(labels)
        napari.run()

print("All checks run!")
