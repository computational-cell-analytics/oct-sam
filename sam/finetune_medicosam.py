import os
from glob import glob

import torch_em
from micro_sam.training import train_sam_for_configuration
from sklearn.model_selection import train_test_split


def get_loaders(input_folder, patch_shape, batch_size, val_size=0.1):
    paths = sorted(glob(os.path.join(input_folder, "*.h5")))
    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    image_key = "image"
    label_key = "labels/original"
    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=25,
    )

    train_loader = torch_em.default_segmentation_loader(
        train_paths, image_key, train_paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=100 * batch_size,
    )
    val_loader = torch_em.default_segmentation_loader(
        val_paths, image_key, val_paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=2 * batch_size,
    )
    return train_loader, val_loader


def finetune_medicosam(input_folder, check):
    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(input_folder, patch_shape, batch_size=8)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    train_sam_for_configuration(
        name="oct-sam-v1", train_loader=train_loader, val_loader=val_loader,
        configuration="A100", with_segmentation_decoder=True,
        model_type="vit_b_medical_imaging",
    )


def main():
    input_folder = "/mnt/lustre-grete/usr/u12086/data/oct/data_20250619_resaved"
    finetune_medicosam(input_folder, check=False)


if __name__ == "__main__":
    main()
