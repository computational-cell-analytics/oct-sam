import argparse
import os

import imageio.v3 as imageio
from micro_sam.sam_annotator import image_series_annotator
from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.precompute_segmentation import fill_gaps_watershed, filter_min_thickness
from oct_tools.segmentation_utils import run_measurement
from tqdm import tqdm


def _postprocess_segmentation(seg, img, min_thickness=5):
    seg = filter_min_thickness(seg, min_thickness=min_thickness)
    seg = fill_gaps_watershed(seg, img)
    return seg


def _precompute_segmentation(images, sam_model_path, output_folder, postprocess=True):
    """Precompute segmentation using micro-sam.
    """

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=sam_model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)

    for i, image in tqdm(enumerate(images), desc="Precompute segmentation", total=len(images)):
        output_path = os.path.join(output_folder, f"seg_{i:05}.tif")
        table_out = os.path.join(output_folder, f"meas_{i:05}.tsv")

        # Init the segmenter for this image.
        segmenter.initialize(image, verbose=False)

        foreground, boundary_distances = segmenter._foreground, segmenter._boundary_distances

        prompts = _derive_prompts_sam(foreground, boundary_distances)
        seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
        if postprocess:
            seg = _postprocess_segmentation(seg, image)

        tab = run_measurement(seg)
        tab.to_csv(table_out, sep="\t", index=False)

        imageio.imwrite(output_path, seg)


def run_annotator(input_path, output_folder, slices, sam_model, precompute_segmentation, postprocess=True):
    image_vol = imageio.imread(input_path)
    images = [image_vol[z] for z in slices]

    if precompute_segmentation:
        _precompute_segmentation(images, sam_model, output_folder, postprocess=postprocess)
    image_series_annotator(
        images, output_folder, model_type="vit_b", checkpoint_path=sam_model, skip_segmented=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Apply micro-sam model on a single or multiple slices of input data."
    )
    parser.add_argument("-i", "--input", required=True, help="Input image.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("-z", "--slices", nargs="+", type=int, required=True, help="Slice(s) in z-direction.")
    parser.add_argument("--model", default="./oct-sam-v3.pt", help="The SAM model trained for OCT data model.")
    parser.add_argument("--precompute_segmentation", action="store_true",
                        help="Pre-compute segmentation using prompts derived from micro-sam prediction.")

    args = parser.parse_args()

    run_annotator(
        args.input, args.output, args.slices, args.model, args.precompute_segmentation,
    )


if __name__ == "__main__":
    main()
