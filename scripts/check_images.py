import os
from glob import glob

import imageio.v3 as imageio
import napari

ROOT = "./data"


images = sorted(glob(os.path.join(ROOT, "/**/*_croppedb.tif"), recursive=True))
masks = sorted(glob(os.path.join(ROOT, "/**/*_maskedb.tif"), recursive=True))
assert len(images) == len(masks)

print("Checking", len(images), "images")

for x, y in zip(images, masks):
    x, y = imageio.imread(x), imageio.imread(y)
    v = napari.Viewer()
    v.add_image(x)
    v.add_labels(y)
    napari.run()
