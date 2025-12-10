import imageio.v3 as imageio
import napari
import numpy as np
import vigra

from skimage.measure import label
from skimage.segmentation import watershed

from oct_tools.segmentation_utils import normalize_sliding_max_2d, merge_overseg


z = 10
im = imageio.imread("../data/350436.tif")[z]
pred = imageio.imread("../data/350436_pred.tif")

foreground, boundaries = pred[:, z]
mask = foreground > 0.5
bd_mask = boundaries > 0.5

directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
# directed_dist = vector_distance_transform(bd_mask)

directed_dist[~mask] = 0
directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
# directed_dist = normalize_rows_sliding_max(directed_dist, window=1)
directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

seed_threshold = 0.6
seeds = label(directed_dist > seed_threshold)
over_seg = watershed(1. - directed_dist, markers=seeds, mask=mask)

# _, seg = cluster_labels_into_bands(over_seg, n_bands=6)
seg = merge_overseg(over_seg, directed_dist, beta=0.5)

v = napari.Viewer()
v.add_image(im)
v.add_image(boundaries)
v.add_labels(bd_mask)
v.add_image(directed_dist)
v.add_labels(seeds)
v.add_labels(over_seg)
v.add_labels(seg)
napari.run()
