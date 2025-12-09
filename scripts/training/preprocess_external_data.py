import os
from glob import glob
from pathlib import Path

import h5py
import imageio.v3 as imageio
import napari
import numpy as np
from tqdm import trange, tqdm

ROOT_OCT5K = "./data/pretrain/OCT5k/data/OCT5k"
ROOT_HCMS = "./data/hcms"
ROOT_DUKE = "./data/2015_BOE_Chiu"


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


def _mat_to_labels(control_pts, shape):
    n_slices, n_surfaces = control_pts.shape
    assert n_slices == shape[0]

    # Determine which surfaces actually contain points.
    nonempty = np.array(
        [[control_pts[i, j].size > 0 for j in range(n_surfaces)]
         for i in range(n_slices)]
    )
    used_surfaces = np.where(nonempty.any(axis=0))[0]

    # Create the segmentation.
    seg = np.zeros(shape,  dtype="uint8")
    _, height, width = shape

    # Regular x-grid in the same coordinate system as the control points (1-based).
    x_grid = np.arange(1, width + 1, dtype=np.float32)

    n_used = len(used_surfaces)
    surface_y = np.empty((n_used, width), dtype=np.float32)

    for s in trange(n_slices, desc="Write borders as label masks"):
        # Interpolate each surface on the regular x-grid
        for idx, j in enumerate(used_surfaces):
            pts = control_pts[s, j]
            if pts.size == 0:
                surface_y[idx, :] = np.nan
                continue

            xs = pts[:, 0].astype(np.float32)
            ys = pts[:, 1].astype(np.float32)

            order = np.argsort(xs)
            xs = xs[order]
            ys = ys[order]

            # 1D interpolation; outside the annotated range, clamp to endpoints
            surface_y[idx, :] = np.interp(x_grid, xs, ys, left=ys[0], right=ys[-1])

        # (Optional) handle NaNs by copying from neighboring slices; omitted here

        # Assume surfaces are already ordered inner->outer (small y -> large y).
        # We create one band between each pair of successive surfaces.
        for b in range(n_used - 1):
            y_top = surface_y[b, :] - 1.0  # convert 1-based to 0-based indices
            y_bot = surface_y[b + 1, :] - 1.0

            # Clip to image bounds
            y_top = np.clip(np.floor(y_top).astype(int), 0, height - 1)
            y_bot = np.clip(np.ceil(y_bot).astype(int), 0, height - 1)

            for x in range(width):
                t = y_top[x]
                btm = y_bot[x]
                if btm <= t:
                    continue
                # Label bands starting from 1
                seg[s, t:btm, x] = b + 1

    return seg


def prepare_hcms():
    import eyepy as ep
    from scipy.io import loadmat

    # output_folder = None
    output_folder = "./pretrain_data/hcms"

    tomograms = glob(os.path.join(ROOT_HCMS, "vol", "*.vol"))
    for i, tomo in enumerate(tomograms):
        fname = os.path.basename(tomo).replace(".vol", ".mat")
        base_name = os.path.splitext(fname)[0]

        if output_folder is not None:
            z = 0
            out_path = os.path.join(output_folder, f"{base_name}_{z:03}.h5")
            if os.path.exists(out_path):
                continue

        print("Processing tomo", i, "/", len(tomograms))
        label_path = os.path.join(ROOT_HCMS, "delineation", fname)
        try:
            masks = loadmat(label_path)["control_pts"]
        except KeyError:
            print("Could not find control points for", tomo)
            continue

        ev = ep.import_heyex_vol(tomo)
        data = ev.data
        labels = _mat_to_labels(masks, data.shape)
        assert labels.shape == data.shape

        if output_folder is None:
            v = napari.Viewer()
            v.add_image(data)
            v.add_labels(labels)
            napari.run()
        else:
            os.makedirs(output_folder, exist_ok=True)
            for z in range(data.shape[0]):
                out_path = os.path.join(output_folder, f"{base_name}_{z:03}.h5")
                with h5py.File(out_path, "a") as f:
                    f.create_dataset("image", data=data[z], compression="gzip")
                    f.create_dataset("masks", data=labels[z], compression="gzip")


def _load_duke_data(data, which_layers="manual1"):
    images = data["images"]  # (H,W,N)
    H, W, N = images.shape
    images_bhw = np.transpose(images, (2, 0, 1))

    # pick layer set
    if which_layers == "manual1":
        layers = data["manualLayers1"]
    elif which_layers == "manual2":
        layers = data["manualLayers2"]
    elif which_layers == "avg_manual":
        layers = np.nanmean(
            np.stack([data["manualLayers1"], data["manualLayers2"]], axis=0),
            axis=0,
        )
    else:
        raise ValueError

    L, W_l, N_l = layers.shape
    assert W_l == W and N_l == N

    labels = np.zeros((N, H, W), np.uint8)

    xs_full = np.arange(W, dtype=np.float64)

    for b in range(N):
        if np.all(np.isnan(layers[:, :, b])):
            continue

        surf_y = np.full((L, W), np.nan, np.float64)

        for s in range(L):
            y = layers[s, :, b]
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue

            xs = xs_full[valid]
            ys = y[valid]

            xmin, xmax = xs.min(), xs.max()

            # interpolate only inside [xmin, xmax]
            mask = (xs_full >= xmin) & (xs_full <= xmax)
            surf_y[s, mask] = np.interp(xs_full[mask], xs, ys)

        # require all surfaces defined
        valid_cols = ~np.isnan(surf_y).any(axis=0)
        cols = np.where(valid_cols)[0]

        if len(cols) == 0:
            continue

        # convert to pixels
        surf_pix = np.round(surf_y[:, cols] - 1).astype(int)
        surf_pix = np.clip(surf_pix, 0, H-1)

        # fill bands
        for band in range(L-1):
            y0 = surf_pix[band]
            y1 = surf_pix[band+1]
            top = np.minimum(y0, y1)
            bot = np.maximum(y0, y1)
            for i, x in enumerate(cols):
                if bot[i] > top[i]:
                    labels[b, top[i]:bot[i]+1, x] = band+1

    return images_bhw, labels


def prepare_duke_dme(cut_labels=True):
    from scipy.io import loadmat

    # output_folder = None
    output_folder = "./pretrain_data/duke_dme"

    files = glob(os.path.join(ROOT_DUKE, "**/*.mat"), recursive=True)
    for ff in tqdm(files, desc="Process files"):
        data = loadmat(ff, struct_as_record=False, squeeze_me=True)
        images, labels = _load_duke_data(data, which_layers="manual1")

        if cut_labels:
            image_planes, label_planes = [], []
            for plane, label_plane in zip(images, labels):
                if label_plane.max() == 0:
                    continue
                bb = np.where(label_plane != 0)
                bb = np.s_[0:plane.shape[0], bb[1].min():bb[1].max() + 1]
                image_planes.append(plane[bb])
                label_planes.append(label_plane[bb])
            try:
                images = np.stack(image_planes)
                labels = np.stack(label_planes)
            except ValueError:
                images, labels = image_planes, label_planes

        if output_folder is None:
            v = napari.Viewer()
            v.add_image(images)
            v.add_labels(labels)
            napari.run()
        else:
            fname = Path(ff).stem
            os.makedirs(output_folder, exist_ok=True)
            for z in range(len(images)):
                out_path = os.path.join(output_folder, f"{fname}_z{z:03}.h5")
                if os.path.exists(out_path):
                    continue
                with h5py.File(out_path, "a") as f:
                    f.create_dataset("image", data=images[z], compression="gzip")
                    f.create_dataset("masks", data=labels[z], compression="gzip")


def main():
    # prepare_oct5k_dataset()
    # prepare_hcms()
    prepare_duke_dme()


if __name__ == "__main__":
    main()
