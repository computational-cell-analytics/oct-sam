import argparse

import magicgui
import napari
import numpy as np
import vigra

from skimage.measure import label
from skimage.segmentation import watershed
from util import load_volume, run_prediction
from util import normalize_sliding_max_2d, merge_overseg


def _run_segmentation(prediction, n_layers, merge):
    foreground, boundaries = prediction
    mask = foreground > 0.5
    bd_mask = boundaries > 0.5

    directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
    directed_dist[~mask] = 0
    directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
    directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

    seed_threshold = 0.6
    seeds = label(directed_dist > seed_threshold)
    over_seg = watershed(1. - directed_dist, markers=seeds, mask=mask)

    if merge:
        seg = merge_overseg(over_seg, directed_dist, beta=0.5)
    else:
        seg = over_seg
    return seg


def segmentation_tool(input_path, prediction_path, model_path):
    tomogram = load_volume(input_path)
    if prediction_path is None:
        print("Run prediction with model", model_path)
        prediction = run_prediction(tomogram, model_path)
    else:
        print("Loading precomputed prediction from", prediction_path)
        prediction = load_volume(prediction_path)

    viewer = napari.Viewer()

    @magicgui.magicgui(call_button="Segment")
    def segment_widget(viewer: napari.Viewer, n_layers: int = 6, merge: bool = True):
        pos = viewer.cursor.position
        z = int(pos[0]) if len(pos) == 3 else int(pos[1])
        seg = _run_segmentation(prediction[:, z], n_layers, merge)
        viewer.layers["segmentation"].data[z] = seg
        viewer.layers["segmentation"].refresh()

    segmentation = np.zeros(tomogram.shape, dtype="uint8")
    viewer.add_image(tomogram)
    viewer.add_image(prediction, name="prediction", visible=False)
    viewer.add_labels(segmentation)

    viewer.window.add_dock_widget(segment_widget)

    napari.run()


def main():
    parser = argparse.ArgumentParser(description="First version of the OCT segmentation tool.")
    parser.add_argument(
        "-i", "--input", required=True, help="The path to the oct file. Can either be a .tiff or .vol file."
    )
    parser.add_argument(
        "-p", "--prediction", help="Optional path to a precomputed network prediction. Must be a .tiff file."
    )
    parser.add_argument(
        "-m", "--model", default="./models/oct-2d-v1.pt", help="The path to the segmentation model."
    )
    args = parser.parse_args()

    segmentation_tool(args.input, args.prediction, args.model)


if __name__ == "__main__":
    main()
