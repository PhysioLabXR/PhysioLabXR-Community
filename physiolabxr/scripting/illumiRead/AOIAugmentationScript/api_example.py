import pickle
from eidl.utils.model_utils import get_subimage_model


if __name__ == '__main__':

    # load image data ###########################################################
    # model and the image data will be downloaded when first used
    # find the best model in result directory
    subimage_handler = get_subimage_model()

    # you can either provide or not provide the source (human) attention as an argument to subimage_handler.compute_perceptual_attention(),
    # if not provided, the model attention will be returned otherwise the perceptual attention will be returned

    rtn = subimage_handler.compute_perceptual_attention('9025_OD_2021_widefield_report', is_plot_results=True, discard_ratio=0.1)
    print(rtn)