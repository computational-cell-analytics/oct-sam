import argparse
import os

import imageio.v3 as imageio
from micro_sam.util import get_sam_model
from micro_sam.sam_annotator import image_series_annotator

from util import _derive_prompts, _segment_from_prompts, _load_model


def _precompute_segmentation(images, model_path, sam_model_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    model = _load_model(model_path)
    predictor = get_sam_model(model_type="vit_b", checkpoint_path=sam_model_path)

    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"seg_{i:05}.tif")
        prompts = _derive_prompts(model, image)
        seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
        imageio.imwrite(output_path, seg)


def run_annotator(input_path, output_folder, slices, model, sam_model, precompute_segmentation):
    image_vol = imageio.imread(input_path)
    images = [image_vol[z] for z in slices]
    if precompute_segmentation:
        _precompute_segmentation(images, model, sam_model, output_folder)
    image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=sam_model, skip_segmented=False
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-z", "--slices", nargs="+", type=int, required=True)
    parser.add_argument("--model", default="./oct-2d-v2.pt")
    parser.add_argument("--sam_model", default="./oct-sam-v1.pt")
    parser.add_argument("--precompute_segmentation", action="store_true")
    args = parser.parse_args()

    run_annotator(args.input, args.output, args.slices, args.model, args.sam_model, args.precompute_segmentation)


if __name__ == "__main__":
    main()
