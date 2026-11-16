import torch_em
from micro_sam.training import train_sam_for_configuration
from micro_sam.util import export_custom_sam_model


def raw_trafo(x):
    x = 255 * torch_em.transform.raw.normalize(x)
    return x


def export_model(model_name):
    export_custom_sam_model(
        f"./checkpoints/{model_name}/best.pt", model_type="vit_b", save_path=f"./{model_name}.pt",
        with_segmentation_decoder=True
    )


def train_oct_sam_model(
    model_name: str,
    train_loader,
    val_loader,
    check: bool = False,
):
    if check:
        from torch_em.util.debug import check_loader
        check_loader(train_loader, n_samples=8)
        check_loader(val_loader, n_samples=8)

    train_sam_for_configuration(
        name=model_name, train_loader=train_loader, val_loader=val_loader,
        configuration="V100", with_segmentation_decoder=True,
        model_type="vit_b_medical_imaging",
        verify_n_labels_in_loader=5,
        n_epochs=40,
        strict_decoder_loading=False,
    )
