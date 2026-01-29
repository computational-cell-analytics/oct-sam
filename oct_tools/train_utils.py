import numpy as np
import torch_em
import vigra
from skimage.measure import regionprops


class BoundaryAndDistanceTransform:
    eps = 1e-7

    def __init__(self):
        self.label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True, ndim=2)

    def compute_normalized_directed_distances(self, mask, boundaries, bb, distances):
        cropped_mask = mask[bb]
        inv_mask = ~cropped_mask

        cropped_boundary_mask = boundaries[bb]
        this_distances = np.abs(vigra.filters.vectorDistanceTransform(cropped_boundary_mask))
        this_distances[inv_mask] = 0

        spatial_axes = tuple(range(mask.ndim))
        this_distances /= (np.abs(this_distances).max(axis=spatial_axes, keepdims=True) + self.eps)

        distances[bb][cropped_mask] = this_distances[cropped_mask]
        return distances

    def __call__(self, labels: np.ndarray) -> np.ndarray:
        fg_and_bd = self.label_transform(labels)
        assert fg_and_bd.shape[0] == 2
        boundaries = fg_and_bd[1].astype("uint32")

        # Compute region properties to derive bounding boxes and centers.
        ndim = labels.ndim
        labels = labels + 1
        props = regionprops(labels)
        bounding_boxes = {
            prop.label: tuple(slice(prop.bbox[i], prop.bbox[i + ndim]) for i in range(ndim)) for prop in props
        }

        # Compute how many distance channels we have.
        n_channels = 2

        # Compute the per object distances.
        distances = np.full(labels.shape + (n_channels,), 1, dtype="float32")
        for prop in props:
            label_id = prop.label
            mask = labels == label_id
            distances = self.compute_normalized_directed_distances(
                mask, boundaries, bounding_boxes[label_id], distances
            )

        # Bring the distance channel to the first dimension.
        to_channel_first = (ndim,) + tuple(range(ndim))
        distances = distances.transpose(to_channel_first)

        return np.concatenate([fg_and_bd, distances], axis=0)
