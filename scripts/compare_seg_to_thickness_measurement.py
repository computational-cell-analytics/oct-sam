import argparse

from oct_tools.analysis.eval_thickness_measurement import eval_thickness_nnunet_multi


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate nnU-Net segmentation by comparing it to manually determined layer thicknesses."
    )

    parser.add_argument("-m", "--measurement", type=str, required=True)
    parser.add_argument("-i", "--nnunet_dir", type=str, required=True,
                        help="Directory with nnU-Net inferences.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output JSON with errors.")

    args = parser.parse_args()

    eval_thickness_nnunet_multi(
        args.measurement,
        args.nnunet_dir,
        args.output,
    )


if __name__ == "__main__":
    main()
