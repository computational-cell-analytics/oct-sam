import argparse
import os
from glob import glob

import eyepy as ep
import napari
import nibabel as nib
import numpy as np
from scipy.io import loadmat
from tqdm import trange, tqdm


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


def prepare_hcms(input_folder, output_folder, pixel_spacing=(3.87, 5.8, 123.6), combine_is_os=True, output_3d=False):
    # The affine matrix defines the spatial orientation and position
    # Default affine assumes the origin is at (0,0,0) and voxel spacing is as specified
    affine = np.eye(4)  # Identity matrix (standard for most cases)
    # µm per voxel (x, y, z)
    affine[0, 0] = pixel_spacing[0]  # x-spacing
    affine[1, 1] = pixel_spacing[1]  # y-spacing
    if output_3d:
        affine[2, 2] = pixel_spacing[2]  # z-spacing

    image_dir = os.path.join(output_folder, "imagesTr")
    label_dir = os.path.join(output_folder, "labelsTr")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    tomograms = glob(os.path.join(input_folder, "vol", "*.vol"))
    tomograms.sort()
    for i, tomo in enumerate(tomograms):
        fname = os.path.basename(tomo).replace(".vol", ".mat")
        base_name = os.path.splitext(fname)[0]

        # get index for output in nnUNet format
        if base_name[:2] == "hc":
            dataset_id = "11"
        elif base_name[:2] == "ms":
            dataset_id = "12"
        scan_id = base_name[2:4]

        print("Processing tomo", i, "/", len(tomograms))
        label_path = os.path.join(input_folder, "delineation", fname)
        try:
            masks = loadmat(label_path)["control_pts"]
        except KeyError:
            print("Could not find control points for", tomo)
            continue

        ev = ep.import_heyex_vol(tomo)
        data = ev.data
        labels = _mat_to_labels(masks, data.shape)
        assert labels.shape == data.shape

        if output_3d:
            nnunet_identifier = f"{dataset_id}{scan_id.zfill(3)}"
            image_path = os.path.join(image_dir, f"oct_{nnunet_identifier}_0000.nii.gz")

            # Create the NIfTI image
            nifti_image = nib.Nifti1Image(data, affine)
            nib.save(nifti_image, image_path)

            # combine inner (label 6) and outer (label 7) photoreceptor segments
            # combine labels 6 and 7, shift label 8 to 7
            if combine_is_os:
                unique_ids = np.unique(labels)[1:]
                assert unique_ids[-1] == 8
                labels[labels == 7] = 6
                labels[labels == 8] = 7

            nifti_label = nib.Nifti1Image(labels, affine)
            label_path = os.path.join(label_dir, f"oct_{nnunet_identifier}.nii.gz")
            nib.save(nifti_label, label_path)
        else:
            for z in range(data.shape[0]):
                nnunet_identifier = f"{dataset_id}{scan_id.zfill(3)}_{str(z).zfill(3)}"
                image_path = os.path.join(image_dir, f"oct_{nnunet_identifier}_0000.nii.gz")

                # Create the NIfTI image
                nifti_image = nib.Nifti1Image(data[z], affine)
                nib.save(nifti_image, image_path)

                # combine inner (label 6) and outer (label 7) photoreceptor segments
                # combine labels 6 and 7, shift label 8 to 7
                label = labels[z]
                if combine_is_os:
                    unique_ids = np.unique(label)[1:]
                    assert unique_ids[-1] == 8
                    label[label == 7] = 6
                    label[label == 8] = 7

                nifti_label = nib.Nifti1Image(label, affine)
                label_path = os.path.join(label_dir, f"oct_{nnunet_identifier}.nii.gz")
                nib.save(nifti_label, label_path)


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


def prepare_duke_dme(input_folder, output_folder, cut_labels=True, pixel_spacing=(3.87, 11.33)):
    # The affine matrix defines the spatial orientation and position
    # Default affine assumes the origin is at (0,0,0) and voxel spacing is as specified
    affine = np.eye(4)  # Identity matrix (standard for most cases)
    # µm per voxel (x, y, z)
    affine[0, 0] = pixel_spacing[0]  # x-spacing
    affine[1, 1] = pixel_spacing[1]  # y-spacing

    image_dir = os.path.join(output_folder, "imagesTr")
    label_dir = os.path.join(output_folder, "labelsTr")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # get index for output in nnUNet format
    dataset_id = "20"

    files = glob(os.path.join(input_folder, "**/*.mat"), recursive=True)
    files.sort()
    for ff in tqdm(files, desc="Process files"):
        data = loadmat(ff, struct_as_record=False, squeeze_me=True)
        images, labels = _load_duke_data(data, which_layers="manual1")

        base_name = ff.split(".mat")[0]
        scan_id = base_name[-2:]

        # extract slice indexes because the volumes are labeled sparsely
        slice_indexes = []
        if cut_labels:
            image_planes, label_planes = [], []
            for num, (plane, label_plane) in enumerate(zip(images, labels)):
                if label_plane.max() == 0:
                    continue
                bb = np.where(label_plane != 0)
                bb = np.s_[0:plane.shape[0], bb[1].min():bb[1].max() + 1]
                image_planes.append(plane[bb])
                label_planes.append(label_plane[bb])
                slice_indexes.append(num)
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
            if len(slice_indexes) == 0:
                slice_indexes = [i for i in range(len(images))]
            for z, slice_index in enumerate(slice_indexes):
                nnunet_identifier = f"{dataset_id}{str(scan_id).zfill(3)}{str(slice_index).zfill(3)}"
                image_path = os.path.join(image_dir, f"oct_{nnunet_identifier}_0000.nii.gz")

                # Create the NIfTI image
                nifti_image = nib.Nifti1Image(images[z], affine)
                nib.save(nifti_image, image_path)

                nifti_label = nib.Nifti1Image(labels[z], affine)
                label_path = os.path.join(label_dir, f"oct_{nnunet_identifier}.nii.gz")
                nib.save(nifti_label, label_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SAM model on all images in a folder. Evaluates data in h5 format."
    )

    parser.add_argument("-i", "--input_dir", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("--output_3d", action="store_true",
                        help="Output 3D data for hcms dataset.")

    parser.add_argument("--dataset", type=str, default="hcms",
                        help="Specifiy dataset. Either 'duke_dme' or 'hcms'. Default: hcms.")

    args = parser.parse_args()

    if args.dataset == "hcms":
        prepare_hcms(
            input_folder=args.input_dir,
            output_folder=args.output_dir,
            output_3d=args.output_3d,
        )
    elif args.dataset == "duke_dme":
        prepare_duke_dme(
            input_folder=args.input_dir,
            output_folder=args.output_dir,
        )
    else:
        raise ValueError("Choose either 'hcms' or 'duke_dme' for the creation of a dataset.")


if __name__ == "__main__":
    main()
