import os
from glob import glob

import torch_em
from micro_sam.training import train_sam_for_configuration
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset

ROOT_DME = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/duke_dme"
ROOT_HCMS = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/pretrain_data/hcms"
ROOT_DOROTHEA = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data"


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


def get_loaders(patch_shape, batch_size, val_size=0.1):
    # pretrain data from public datasets
    paths = sorted(glob(os.path.join(ROOT_DME, "*.h5"))) +\
        sorted(glob(os.path.join(ROOT_HCMS, "*.h5")))
    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    image_key = "image"
    label_key = "masks"

    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
        min_size=10,
    )

    train_dataset_01 = torch_em.default_segmentation_dataset(
        train_paths, image_key, train_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        raw_transform=raw_trafo,
    )

    val_dataset_01 = torch_em.default_segmentation_dataset(
        val_paths, image_key, val_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        raw_transform=raw_trafo,
    )

    # training data from Dorothea
    paths = []
    image_key = "image"
    label_key = "labels/edit_v3"
    input_folders = [
        f"{ROOT_DOROTHEA}/standard_20250619",
        f"{ROOT_DOROTHEA}/standard_20251126",
        f"{ROOT_DOROTHEA}/standard_20251215",
        f"{ROOT_DOROTHEA}/standard_20260105",
    ]
    for input_folder in input_folders:
        paths.extend(sorted(glob(os.path.join(input_folder, "*.h5"))))
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    train_dataset_02 = torch_em.default_segmentation_dataset(
        train_paths, image_key, train_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        raw_transform=raw_trafo
    )

    val_dataset_02 = torch_em.default_segmentation_dataset(
        val_paths, image_key, val_paths, label_key,
        patch_shape=patch_shape,
        label_transform=label_transform, is_seg_dataset=True,
        raw_transform=raw_trafo,
    )

    # combine training data
    train_dataset = ConcatDataset([train_dataset_01, train_dataset_02])
    val_dataset = ConcatDataset([val_dataset_01, val_dataset_02])

    train_loader = torch_em.get_data_loader(train_dataset, batch_size, num_workers=8)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size, num_workers=8)

    return train_loader, val_loader


def pretrain_medicosam(check):
    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(patch_shape, batch_size=1)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    train_sam_for_configuration(
        name="oct-sam-trained-v1", train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type="vit_b_medical_imaging",
        verify_n_labels_in_loader=5,
        n_epochs=40,
    )


def export_pretrained_model():
    from micro_sam.util import export_custom_sam_model
    export_custom_sam_model(
        "./checkpoints/oct-sam-trained-v1/best.pt", model_type="vit_b", save_path="./oct-sam-trained-v1.pt",
        with_segmentation_decoder=True
    )


def main():
    """Train medico-sam directly on all datasets - two public datasets and the data from Dorothea.
    """
    pretrain_medicosam(check=False)
    export_pretrained_model()


if __name__ == "__main__":
    main()
