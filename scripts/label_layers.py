import argparse
import json

import h5py
import imageio.v3 as imageio
import numpy as np

from oct_tools.layer_information import identify_layers


def label_layers(
    input_path: str,
    output_path: str,
):
    """Label layers of segmentation.

    Args:
        input_path: File path to 2D segmentation of OCT data.
        output_path: File path to save metrics as table in TSV format.
    """
    if ".h5" in input_path:
        seg = np.array(h5py.File(input_path, "r")["labels"]["original"])
    else:
        seg = imageio.imread(input_path)
    label_dict = identify_layers(seg)

    if output_path is not None:
        with open(output_path, "w") as f:
            json.dump(label_dict, f, indent='\t', separators=(',', ': '))
    else:
        print(label_dict)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate oct-metrics for 2D segmentation."
    )
    parser.add_argument("-i", "--input", required=True, help="Input segmentation.")
    parser.add_argument("-o", "--output", default=None,
                        help="Output path for JSON.")

    args = parser.parse_args()

    label_layers(
        args.input, args.output
    )


if __name__ == "__main__":
    main()
