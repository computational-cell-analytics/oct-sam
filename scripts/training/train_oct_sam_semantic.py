import argparse
import os
from glob import glob

import torch_em
from torch.utils.data import ConcatDataset

from oct_tools.train_utils import export_model, raw_trafo, train_oct_sam_model

ROOT_DIR = "/mnt/vast-nhr/projects/nim00007/data/mace/oct-data/semantic_training_medicosam"


def get_loaders(patch_shape, batch_size):
    # pretrain data from public datasets
    train_paths = sorted(glob(os.path.join(ROOT_DIR, "duke_dme_train", "*.h5"))) +\
        sorted(glob(os.path.join(ROOT_DIR, "hcms_train", "*.h5")))
    val_paths = sorted(glob(os.path.join(ROOT_DIR, "duke_dme_val", "*.h5"))) +\
        sorted(glob(os.path.join(ROOT_DIR, "hcms_val", "*.h5")))

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

    # training data from the UMG-RP dataset
    image_key = "image"
    label_key = "labels/edit_v3"
    train_paths = sorted(glob(os.path.join(ROOT_DIR, "umg_rp_train", "*.h5")))
    val_paths = sorted(glob(os.path.join(ROOT_DIR, "umg_rp_val", "*.h5")))

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


def main():
    parser = argparse.ArgumentParser(
        description="Train OCT-SAM for semantic segmentation on both public and the UMG-RP datset."
    )
    parser.add_argument("-m", "--model", type=str, default="oct-sam-semantic",
                        help="Model name of pretrained model.")
    args = parser.parse_args()

    patch_shape = (384, 992)
    train_loader, val_loader = get_loaders(patch_shape, batch_size=1)
    train_oct_sam_model(
        model_name=args.model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        check=False,
    )
    export_model(args.model_name)


if __name__ == "__main__":
    main()
