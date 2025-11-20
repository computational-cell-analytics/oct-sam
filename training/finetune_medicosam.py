import os
from glob import glob

import torch_em
from micro_sam.training import train_sam_for_configuration
from sklearn.model_selection import train_test_split


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


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
        n_samples=100 * batch_size, raw_transform=raw_trafo,
    )
    val_loader = torch_em.default_segmentation_loader(
        val_paths, image_key, val_paths, label_key,
        batch_size=batch_size, patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        n_samples=2 * batch_size, raw_transform=raw_trafo,
    )
    return train_loader, val_loader


def finetune_medicosam(input_folder, check):
    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(input_folder, patch_shape, batch_size=1)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    train_sam_for_configuration(
        name="oct-sam-v1", train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type="vit_b_medical_imaging",
        verify_n_labels_in_loader=5,
    )


def export_finetuned_model():
    from micro_sam.util import export_custom_sam_model
    export_custom_sam_model(
        "./checkpoints/oct-sam-v1/best.pt", model_type="vit_b", save_path="./oct-sam-v1.pt",
        with_segmentation_decoder=True
    )


def main():
    input_folder = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data_v2"
    finetune_medicosam(input_folder, check=False)
    export_finetuned_model()


if __name__ == "__main__":
    main()
