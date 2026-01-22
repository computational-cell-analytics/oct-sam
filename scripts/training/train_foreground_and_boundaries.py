import os
from glob import glob

import torch_em
from sklearn.model_selection import train_test_split
from torch_em.model import UNet2d
from torch.utils.data import ConcatDataset

ROOT_DME = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/duke_dme"
ROOT_HCMS = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/hcms"
ROOT_DOROTHEA = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data"


def get_loaders(patch_shape, batch_size, val_size=0.1):

    # pretrain data from public datasets
    paths = sorted(glob(os.path.join(ROOT_DME, "*.h5"))) +\
        sorted(glob(os.path.join(ROOT_HCMS, "*.h5")))
    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    image_key = "image"
    label_key = "masks"

    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True, ndim=2)

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


def train_boundaries(check=False):
    model = UNet2d(in_channels=1, out_channels=2, initial_features=32, final_activation="Sigmoid")

    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(patch_shape, batch_size=8)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    trainer = torch_em.default_segmentation_trainer(
        name="oct-boundary-foreground",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-4,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
    )
    trainer.fit(iterations=int(2.5e4))


def main():
    train_boundaries(check=False)


if __name__ == "__main__":
    main()
