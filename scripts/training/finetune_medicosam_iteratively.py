import os
from glob import glob

import torch_em
from micro_sam.training import train_sam_for_configuration


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


def _get_loaders(train_dir, val_dir, patch_shape, batch_size):
    image_key = "image"
    label_key = "labels/edit_v3"

    train_paths = sorted(glob(os.path.join(train_dir, "*.h5")))
    val_paths = sorted(glob(os.path.join(val_dir, "*.h5")))

    print("Intialize data loaders with:")
    print(len(train_paths), "training images.")
    print(len(val_paths), "val images.")

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


def finetune_medicosam(
    train_dir,
    val_dir,
    model_name,
    checkpoint_path,
):
    patch_shape = (384, 992)
    model_type = "vit_b"
    train_loader, val_loader = _get_loaders(train_dir, val_dir, patch_shape, batch_size=1)

    train_sam_for_configuration(
        name=model_name, train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type=model_type, checkpoint_path=checkpoint_path,
        verify_n_labels_in_loader=5, early_stopping=30,
        n_epochs=250,
    )


def export_finetuned_model(model_name):
    from micro_sam.util import export_custom_sam_model
    export_custom_sam_model(
        f"./checkpoints/{model_name}/best.pt", model_type="vit_b", save_path=f"./{model_name}.pt",
        with_segmentation_decoder=True
    )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dir", type=str, required=True,
                        help="Train data.")
    parser.add_argument("-v", "--val_dir", type=str, required=True,
                        help="Validation data.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Model name of finetuned model.")
    parser.add_argument("-c", "--checkpoint", type=str, required=True,
                        help="Model checkpoint of pretrained model.")

    args = parser.parse_args()
    finetune_medicosam(
        args.train_dir,
        args.val_dir,
        model_name=args.model,
        checkpoint_path=args.checkpoint,
    )
    export_finetuned_model(args.model)


if __name__ == "__main__":
    main()
