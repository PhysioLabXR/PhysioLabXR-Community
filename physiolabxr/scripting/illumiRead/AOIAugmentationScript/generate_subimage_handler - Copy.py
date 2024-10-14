# import pickle
# from eidl.utils.model_utils import get_subimage_model
#
# from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
# from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import ImageInfo, \
#     get_image_on_screen_shape
# import pickle
#
# current_image_info = None
#
# if __name__ == '__main__':
#
#     # load image data ###########################################################
#     # model and the image data will be downloaded when first used
#     # find the best model in result directory
#
#     image_id = '9025_OD_2021_widefield_report'
#
#
#     # load subimage_handler from pickle
#     with open('subimage_handler.pkl', 'rb') as f:
#         subimage_handler = pickle.load(f)
#
#     # subimage_handler = get_subimage_model()
#     #
#     # # # pickle subimage_handler
#     with open('subimage_handler.pkl', 'wb') as f:
#         pickle.dump(subimage_handler, f)
#
#     # you can either provide or not provide the source (human) attention as an argument to subimage_handler.compute_perceptual_attention(),
#     # if not provided, the model attention will be returned otherwise the perceptual attention will be returned
#
#     image_attention_info_dict = subimage_handler.compute_perceptual_attention(image_id, is_plot_results=False, discard_ratio=0)
#     current_image_info_dict = subimage_handler.image_data_dict[image_id]
#
#     image_info_dict = {**image_attention_info_dict, **current_image_info_dict}
#
#     current_image_info = ImageInfo(**image_info_dict)
#
#
#     image_on_screen_shape = get_image_on_screen_shape(
#         original_image_width=current_image_info.original_image.shape[1],
#         original_image_height=current_image_info.original_image.shape[0],
#         image_width=AOIAugmentationConfig.image_on_screen_max_width,
#         image_height=AOIAugmentationConfig.image_on_screen_max_height,
#     )
#
#
#     # get sub images
#     sub_images_rgba = current_image_info.get_sub_images_rgba(normalized=True, plot_results=True)
#
#     print('image_on_screen_shape', image_on_screen_shape)
#

import pickle

import numpy as np

from eidl.utils.model_utils import get_subimage_model


if __name__ == '__main__':
    subimage_handler = get_subimage_model(n_jobs=20)
    with open('subimage_handler_0.0.22.pkl', 'wb') as f:
        pickle.dump(subimage_handler, f)
