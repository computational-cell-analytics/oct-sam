import os
from glob import glob

import imageio.v3 as imageio

ROOT = "/mnt/lustre-grete/usr/u12086/data/oct/data_20250625"


def convert_volumes():
    import eyepy as ep

    output_folder = os.path.join(ROOT, "converted")
    os.makedirs(output_folder, exist_ok=True)

    print("Start conversion ...")
    files = glob(os.path.join(ROOT, "volumes", "*.vol"))
    for ff in files:
        data = ep.import_heyex_vol(ff).data
        output_path = os.path.join(output_folder, os.path.basename(ff).replace(".vol", ".tif"))
        imageio.imwrite(output_path, data)


def check_converted():
    import napari

    files = glob(os.path.join(ROOT, "converted", "*.tif"))
    for ff in files:
        data = imageio.imread(ff)
        v = napari.Viewer()
        v.add_image(data)
        napari.run()


def main():
    # convert_volumes()
    check_converted()


if __name__ == "__main__":
    main()
