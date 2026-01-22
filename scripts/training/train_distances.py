import os
from glob import glob

import numpy as np
import torch_em
import vigra
from sklearn.model_selection import train_test_split
from skimage.measure import regionprops
from torch_em.model import UNet2d
from torch.utils.data import ConcatDataset

ROOT_DME = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/duke_dme"
ROOT_HCMS = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/hcms"
ROOT_DOROTHEA = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data"


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


def get_loaders(patch_shape, batch_size, val_size=0.1):

    # pretrain data from public datasets
    paths = sorted(glob(os.path.join(ROOT_DME, "*.h5"))) +\
        sorted(glob(os.path.join(ROOT_HCMS, "*.h5")))
    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    image_key = "image"
    label_key = "masks"

    label_transform = BoundaryAndDistanceTransform()

    train_dataset_01 = torch_em.default_segmentation_dataset(
        train_paths, image_key, train_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=100 * batch_size,
    )

    val_dataset_01 = torch_em.default_segmentation_dataset(
        val_paths, image_key, val_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=2 * batch_size,
    )

    # training data from Dorothea
    paths = []
    image_key = "image"
    label_key = "labels/edit_v1"
    input_folders = [
        f"{ROOT_DOROTHEA}/20250619",
        f"{ROOT_DOROTHEA}/20251126",
        f"{ROOT_DOROTHEA}/20251215",
        f"{ROOT_DOROTHEA}/20260105",
    ]
    for input_folder in input_folders:
        paths.extend(sorted(glob(os.path.join(input_folder, "*.h5"))))
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    train_dataset_02 = torch_em.default_segmentation_dataset(
        train_paths, image_key, train_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=100 * batch_size,
    )

    val_dataset_02 = torch_em.default_segmentation_dataset(
        val_paths, image_key, val_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=2 * batch_size,
    )

    # combine training data
    train_dataset = ConcatDataset([train_dataset_01, train_dataset_02])
    val_dataset = ConcatDataset([val_dataset_01, val_dataset_02])

    train_loader = torch_em.get_data_loader(train_dataset, batch_size, num_workers=8)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size, num_workers=8)

    return train_loader, val_loader


def train_distances(check=False):
    model = UNet2d(in_channels=1, out_channels=4, initial_features=32, final_activation="Sigmoid")

    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(patch_shape, batch_size=8)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    trainer = torch_em.default_segmentation_trainer(
        name="oct-boundary-distances-all",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
    )
    trainer.fit(iterations=int(5e4))


# TODO predict a better distance representation:
# - normalized (directed) distance to center line / skeleton?
# - distance exponent?
# - mask out the BG distances
# or just predict the centerline.
def main():
    train_distances(check=False)


if __name__ == "__main__":
    main()
