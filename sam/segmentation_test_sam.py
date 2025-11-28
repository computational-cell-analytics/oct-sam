import imageio.v3 as imageio
import napari

from micro_sam.instance_segmentation import get_amg, get_predictor_and_decoder
from util import _derive_prompts_sam, _segment_from_prompts, _filter_prompts


def test_segmentation_with_sam(image, model_path):
    # Load the trained sam model.
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b", checkpoint_path=model_path)

    # Create the segmenter.
    segmenter = get_amg(predictor, is_tiled=False, decoder=decoder)
    # Init the segmenter for this image.
    segmenter.initialize(image)
    # Run the standard microSAM segmentation function with the most important
    # hyperparameters.
    segmentation = segmenter.generate(
        center_distance_threshold=0.5,
        boundary_distance_threshold=0.5,
        distance_smoothing=0.6,
        min_size=15,
        output_mode=None,
    )

    # Get the intermediate predictions -- we can use them for prompt generation,
    # or try to improve the default segmentation through better hyperparams.
    # For reference how prompts were generated for the U-Net output, see:
    # https://github.com/computational-cell-analytics/oct-analysis/blob/master/sam/segmentation_test.py#L17-L18
    foreground, center_distances, boundary_distances =\
        segmenter._foreground, segmenter._center_distances, segmenter._boundary_distances

    prompts = _derive_prompts_sam(foreground, boundary_distances)
    filtered_prompts = _filter_prompts(prompts)
    segmentation = _segment_from_prompts(predictor, image, filtered_prompts, min_size=150)

    v = napari.Viewer()
    v.add_image(image)
    v.add_image(foreground)
    v.add_image(center_distances)
    v.add_image(boundary_distances)
    v.add_labels(segmentation)
    v.add_points(prompts)
    v.add_points(filtered_prompts)
    napari.run()


def main():
    # Update this to read image / model form somewhere else.
    z = 15
    image = imageio.imread("../data/350436.tif")[z]
    model_path = "./oct-sam-v3.pt"

    test_segmentation_with_sam(image, model_path)


if __name__ == "__main__":
    main()
