import os

import imageio.v3 as imageio
import napari

from segmentation_tool_v2 import _run_segmentation
from oct_tools.segmentation_utils import run_measurement

z = 10

im = imageio.imread("../data/350436.tif")[z]
seg_path = "../data/seg_z10.tif"
if os.path.exists(seg_path):
    seg = imageio.imread(seg_path)
else:
    pred = imageio.imread("../data/350436_pred2.tif")[:, z]
    seg = _run_segmentation(pred, merge=True, seed_threshold=0.6)
    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(seg)
    napari.run()

tab = run_measurement(seg)
breakpoint()

v = napari.Viewer()
v.add_image(im)
v.add_labels(seg)
napari.run()
