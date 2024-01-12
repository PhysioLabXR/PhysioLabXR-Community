import pickle
from eidl.utils.model_utils import get_subimage_model

from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationConfig import PracticeBlockImages, \
    TestBlockImages
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import ImageInfo, \
    get_image_on_screen_shape
import pickle
import numpy as np
import os
import cv2

current_image_info = None

if __name__ == '__main__':



    practice_image_dir = 'data/experiment_images/practice'
    test_image_dir = 'data/experiment_images/test'
    sub_image_handler_path = 'data/subimage_handler.pkl'


    # load subimage_handler from pickle
    with open(sub_image_handler_path, 'rb') as f:
        subimage_handler = pickle.load(f)


    for image_id in PracticeBlockImages:
        original_image = subimage_handler.image_data_dict[image_id]['original_image']
        # save the original image
        cv2.imwrite(os.path.join(practice_image_dir, image_id + '.png'), original_image)

    for image_id in TestBlockImages:
        original_image = subimage_handler.image_data_dict[image_id]['original_image']
        # save the original image
        cv2.imwrite(os.path.join(test_image_dir, image_id + '.png'), original_image)

    # for image_id in subimage_handler.image_data_dict:
    #     original_image = subimage_handler.image_data_dict[image_id]['original_image']
    #     # save the original image
    #     cv2.imwrite(os.path.join(practice_image_dir, image_id + '.png'), original_image)
    #     cv2.imwrite(os.path.join(test_image_dir, image_id + '.png'), original_image)

