import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari

from tqdm import tqdm

ROOT_OCT5K = "./data/pretrain/OCT5k/data/OCT5k"


def _view_masks(im_path, masks):
    image = imageio.imread(im_path)
    v = napari.Viewer()
    v.add_image(image)
    for name, data in masks.items():
        v.add_labels(data, name=name)
    napari.run()


def _save_masks(im_path, masks, out_path):
    os.makedirs(os.path.split(out_path)[0], exist_ok=True)
    with h5py.File(out_path, mode="a") as f:
        f.create_dataset("image", data=imageio.imread(im_path), compression="gzip")
        for name, data in masks.items():
            f.create_dataset(name, data=data, compression="gzip")


def _load_mask(mask_path):
    mask = imageio.imread(mask_path).copy()
    mask[mask == mask.max()] = 0
    return mask


def _process_masks(image_root, *mask_roots, out_root=None):
    mask_root, additional_mask_roots = mask_roots[0], mask_roots[1:]

    files = glob(os.path.join(mask_root, "**/*png"), recursive=True)
    print("Number of files:", len(files))

    for ff in tqdm(files, desc="Process OCT5K masks"):
        rel_path = os.path.relpath(ff, mask_root)
        im_path = os.path.join(image_root, rel_path)
        assert os.path.exists(im_path), im_path

        masks = {"masks1": _load_mask(ff)}
        for i, root in enumerate(additional_mask_roots, 2):
            mask_path = os.path.join(root, rel_path)
            masks[f"masks{i}"] = _load_mask(mask_path)

        if out_root is None:
            _view_masks(im_path, masks)
        else:
            out_name = rel_path.replace(" ", "").replace("(", "_").replace(")", "")
            out_name = f"{os.path.splitext(out_name)[0]}.h5"
            out_path = os.path.join(out_root, out_name)
            _save_masks(im_path, masks, out_path)


def prepare_oct5k_dataset():
    if False:
        image_root = os.path.join(ROOT_OCT5K, "Images/Images_Automatic")
        automatic_masks = os.path.join(ROOT_OCT5K, "Masks/Masks_Automatic/Grading")
        _process_masks(image_root, automatic_masks, out_root="./pretrain_data/oct5k/automatic")
        return

    image_root = os.path.join(ROOT_OCT5K, "Images/Images_Manual")
    masks1 = os.path.join(ROOT_OCT5K, "Masks/Masks_Manual/Grading_1")
    masks2 = os.path.join(ROOT_OCT5K, "Masks/Masks_Manual/Grading_2")
    masks3 = os.path.join(ROOT_OCT5K, "Masks/Masks_Manual/Grading_3")
    _process_masks(image_root, masks1, masks2, masks3, out_root="./pretrain_data/oct5k/manual")


def prepare_hcms():
    pass


def prepare_duke_dme():
    pass


def main():
    # prepare_oct5k_dataset()
    prepare_hcms()


if __name__ == "__main__":
    main()
