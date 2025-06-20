import os
from glob import glob

import torch_em
from sklearn.model_selection import train_test_split
from torch_em.model import UNet2d


# Also create test split?
def get_loaders(input_folder, patch_shape, batch_size, val_size=0.1):
    paths = sorted(glob(os.path.join(input_folder, "*.h5")))
    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    image_key = "image"
    label_key = "labels/original"

    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True, ndim=2)

    train_loader = torch_em.default_segmentation_loader(
        train_paths, image_key, train_paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True
    )
    val_loader = torch_em.default_segmentation_loader(
        val_paths, image_key, val_paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True
    )
    return train_loader, val_loader


def train_boundaries(input_folder, check=False):
    model = UNet2d(in_channels=1, out_channels=2, initial_features=32, final_activation="Sigmoid")

    patch_shape = (374, 998)
    train_loader, val_loader = get_loaders(input_folder, patch_shape, batch_size=8)

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
        log_image_interval=100
    )
    trainer.fit(iterations=int(1e5))


def main():
    input_folder = "../data/data_20250619_resaved"
    train_boundaries(input_folder, check=False)


if __name__ == "__main__":
    main()
