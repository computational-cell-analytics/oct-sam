import imageio.v3 as imageio
import napari

from micro_sam.util import get_sam_model
from util import _derive_prompts, _segment_from_prompts, _load_model


def main():
    z = 15
    im = imageio.imread("../data/350436.tif")[z]
    model_path = "./oct-2d-v2.pt"
    sam_path = "./oct-sam-v1.pt"

    model = _load_model(model_path)
    predictor = get_sam_model(model_type="vit_b", checkpoint_path=sam_path)

    prompts = _derive_prompts(model, im)
    segmentation = _segment_from_prompts(predictor, im, prompts, min_size=150)

    v = napari.Viewer()
    v.add_image(im)
    # v.add_image(pred)
    v.add_labels(segmentation)
    v.add_points(prompts)
    napari.run()


if __name__ == "__main__":
    main()
