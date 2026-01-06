import os
from glob import glob

import torch_em
from micro_sam.training import train_sam_for_configuration
from sklearn.model_selection import train_test_split


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


def _get_loaders(version, patch_shape, batch_size, val_size=0.1):
    if version == 1:
        input_folder = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20250619"
        paths = sorted(glob(os.path.join(input_folder, "*.h5")))
    elif version in (2, 3):
        input_folders = [
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20250619",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20251126",
        ]
        paths = []
        for input_folder in input_folders:
            paths.extend(sorted(glob(os.path.join(input_folder, "*.h5"))))
    elif version == 4:
        input_folders = [
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20250619",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20251126",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20251215",
        ]
        paths = []
        for input_folder in input_folders:
            paths.extend(sorted(glob(os.path.join(input_folder, "*.h5"))))
    elif version == 5:
        input_folders = [
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20250619",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20251126",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20251215",
            "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/training_data/20260105",
        ]
        paths = []
        for input_folder in input_folders:
            paths.extend(sorted(glob(os.path.join(input_folder, "*.h5"))))
    else:
        raise ValueError(f"Version {version} not yet supported.")

    assert len(paths) > 0
    train_paths, val_paths = train_test_split(paths, test_size=val_size, random_state=42)

    print("Intialize data loaders with:")
    print(len(train_paths), "training images.")
    print(len(val_paths), "val images.")

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


def _get_initialization(version):
    if version in (1, 2):
        model_type = "vit_b_medical_imaging"
        checkpoint_path = None
    elif version in (3, 4, 5):
        model_type = "vit_b"
        checkpoint_path = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/models/oct-sam-pretrained-v1.pt"
    else:
        raise ValueError(f"Version {version} not yet supported.")
    return model_type, checkpoint_path


def finetune_medicosam(version, check):
    patch_shape = (384, 992)
    train_loader, val_loader = _get_loaders(version, patch_shape, batch_size=1)

    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    model_type, checkpoint_path = _get_initialization(version)

    train_sam_for_configuration(
        name=f"oct-sam-v{version}", train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type=model_type, checkpoint_path=checkpoint_path,
        verify_n_labels_in_loader=5, early_stopping=30,
        n_epochs=250,
    )


def export_finetuned_model(version):
    from micro_sam.util import export_custom_sam_model
    export_custom_sam_model(
        f"./checkpoints/oct-sam-v{version}/best.pt", model_type="vit_b", save_path=f"./oct-sam-v{version}.pt",
        with_segmentation_decoder=True
    )


# The version determines which data is used for training and how the model is initialized.
# v1: trained on 20250619; initialized with MedicoSAM
# v2: trained on 20250619, 20251126; initialized with MedicoSAM
# v3: trained on 20250619, 20251126; initialized with pretraiend OCT model
# v4: trained on 20250619, 20251126, 20251215; initialized with pretraiend OCT model
# v5: trained on 20250619, 20251126, 20251215, 20260105; initialized with pretraiend OCT model
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--version", type=int, required=True)
    parser.add_argument("-c", "--check", action="store_true")
    args = parser.parse_args()
    finetune_medicosam(args.version, check=args.check)
    export_finetuned_model(args.version)


if __name__ == "__main__":
    main()
