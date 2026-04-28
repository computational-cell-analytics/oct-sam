import argparse
import napari

import imageio.v3 as imageio

SCALE = (140, 6.58, 6.58)


def display_tomogram(input_path):
    tomogram = imageio.imread(input_path)

    v = napari.Viewer()
    v.add_image(tomogram, scale=SCALE)

    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"     # unit to display

    napari.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()
    display_tomogram(args.input)


if __name__ == "__main__":
    main()
