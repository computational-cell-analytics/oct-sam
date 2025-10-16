import argparse
from pathlib import Path

import magicgui
import napari
import numpy as np
import pandas as pd
import vigra

from skimage.measure import label
from skimage.segmentation import watershed
from util import load_volume, run_prediction, run_measurement
from util import normalize_sliding_max_2d, merge_overseg


def _run_segmentation(prediction, merge, seed_threshold):
    foreground, boundaries = prediction[0], prediction[1]
    mask = foreground > 0.5
    bd_mask = boundaries > 0.5

    directed_dist = vigra.filters.vectorDistanceTransform(bd_mask.astype("float32"))
    directed_dist[~mask] = 0
    directed_dist = np.abs(directed_dist.transpose((2, 0, 1)))[0]
    directed_dist = normalize_sliding_max_2d(directed_dist, window_y=1, window_x=255)

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
        prediction = run_prediction(tomogram, model_path, out_channels=2 if "v1" in model_path else 4)
    else:
        print("Loading precomputed prediction from", prediction_path)
        prediction = load_volume(prediction_path)

    viewer = napari.Viewer()

    @magicgui.magicgui(call_button="Segment")
    def segment_widget(viewer: napari.Viewer, seed_threshold: float = 0.6, merge: bool = True):
        pos = viewer.cursor.position
        z = int(pos[0]) if len(pos) == 3 else int(pos[1])
        seg = _run_segmentation(prediction[:, z], seed_threshold=seed_threshold, merge=merge)
        viewer.layers["segmentation"].data[z] = seg
        viewer.layers["segmentation"].refresh()

    @magicgui.magicgui(call_button="Measure")
    def measure_widget(viewer: napari.Viewer, table_path: Path):
        pos = viewer.cursor.position
        z = int(pos[0]) if len(pos) == 3 else int(pos[1])
        seg = viewer.layers["segmentation"].data[z]
        if seg.max() == 0:
            print("The segmentation in slice", z, "is empty. Please click 'segment' first.")
            return

        this_measurements = run_measurement(seg)
        this_measurements["z-slice"] = [z] * len(this_measurements)
        this_measurements["layer_name"] = [" "] * len(this_measurements)

        out_path = table_path.with_suffix(".xlsx")
        if out_path.exists():
            measurements = pd.read_excel(out_path)
            measurements = pd.concat([measurements, this_measurements])
        else:
            measurements = this_measurements
        measurements.to_excel(out_path, index=False)

    segmentation = np.zeros(tomogram.shape, dtype="uint8")
    viewer.add_image(tomogram)
    viewer.add_image(prediction, name="prediction", visible=False)
    viewer.add_labels(segmentation)

    viewer.window.add_dock_widget(segment_widget)
    viewer.window.add_dock_widget(measure_widget)

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
        "-m", "--model", default="./models/oct-2d-v2.pt", help="The path to the segmentation model."
    )
    args = parser.parse_args()

    segmentation_tool(args.input, args.prediction, args.model)


if __name__ == "__main__":
    main()
