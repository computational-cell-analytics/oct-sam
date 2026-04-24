import argparse
import multiprocessing as mp
import os
from concurrent import futures

import h5py
import imageio.v3 as imageio
import numpy as np

from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from torch_em.util.segmentation import watershed_from_center_and_boundary_distances
from oct_tools.postprocessing import postprocess_segmentation
from oct_tools.precompute_segmentation import _derive_prompts_sam, _segment_from_prompts
from oct_tools.metric_utils import run_measurement
from tqdm import tqdm


def run_segmentation(input_path, output_folder, sam_model_path, slices=None, postprocess=True, use_prompts=True):
    if ".h5" in input_path:
        image_vol = [np.array(h5py.File(input_path, "r")["image"])]
    else:
        image_vol = imageio.imread(input_path)
        image_vol = [image_vol[z] for z in range(image_vol.shape[0])]

    if slices is None:
        slices = [i for i in range(len(image_vol))]
    os.makedirs(output_folder, exist_ok=True)

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=sam_model_path)

    def segment_image(slice_id):
        image = image_vol[slice_id]
        output_path = os.path.join(output_folder, f"seg_{slice_id:05}.tif")
        table_out = os.path.join(output_folder, f"meas_{slice_id:05}.tsv")

        # Init the segmenter for this image.
        segmenter.initialize(image, verbose=False)

        foreground = segmenter._foreground
        boundary_distances = segmenter._boundary_distances
        center_distances = segmenter._center_distances

        if use_prompts:
            prompts = _derive_prompts_sam(foreground, boundary_distances)
            if len(prompts) != 0:
                seg = _segment_from_prompts(predictor, image, prompts, min_size=150)
                if postprocess:
                    seg = postprocess_segmentation(seg, image)

                tab = run_measurement(seg)
                tab.to_csv(table_out, sep="\t", index=False)

                imageio.imwrite(output_path, seg)

        else:
            seg = watershed_from_center_and_boundary_distances(center_distances, boundary_distances, foreground)
            if postprocess:
                seg = postprocess_segmentation(seg, image)

            tab = run_measurement(seg)
            tab.to_csv(table_out, sep="\t", index=False)

            imageio.imwrite(output_path, seg)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)
    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(segment_image, slices), total=len(slices)))


def main():
    parser = argparse.ArgumentParser(
        description="Apply SAM model on a single or multiple slices of input data."
    )
    parser.add_argument("-i", "--input", required=True, help="Input image.")
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("-z", "--slices", nargs="+", type=int, help="Slice(s) in z-direction.")
    parser.add_argument("--model", default="./oct-sam-v3.pt", help="The SAM model trained for OCT data model.")
    parser.add_argument("--postprocess", action="store_true",
                        help="Post-process segmentation.")
    parser.add_argument("--no_prompts", action="store_true",
                        help="Do not use two-phase prediction with prompts but only single prediction.")

    args = parser.parse_args()

    run_segmentation(
        args.input, args.output, args.model, args.slices, args.postprocess, not args.no_prompts,
    )


if __name__ == "__main__":
    main()
