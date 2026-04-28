import argparse
import os

import imageio.v3 as imageio
import mrcfile
import eyepy as ep


def main():
    parser = argparse.ArgumentParser()
    parser.add_arument("-i", "--input_path", required=True)
    parser.add_arument("-o", "--output_path", required=True)
    args = parser.parse_args()

    data = ep.import_heyex_vol(args.input_path).data

    output_path = args.output_path
    os.makedirs(os.path.split(output_path)[0], exist_ok=True)
    ext = os.path.splitext(output_path)[1]

    if ext == ".tif":
        imageio.imwrite(output_path, data)
    elif ext in (".mrc", ".raw"):
        with mrcfile.new("output.mrc", overwrite=True) as mrc:
            mrc.set_data(data)


if __name__ == "__main__":
    main()
