import pickle
from eidl.utils.model_utils import get_subimage_model

from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import ImageInfo, \
    get_image_on_screen_shape
import pickle

current_image_info = None

if __name__ == '__main__':

    # load image data ###########################################################
    # model and the image data will be downloaded when first used
    # find the best model in result directory
    sub_image_handler_path = 'data/subimage_handler.pkl'


    # load subimage_handler from pickle
    with open(sub_image_handler_path, 'rb') as f:
        subimage_handler = pickle.load(f)

    for image_id in subimage_handler.image_data_dict:
        print(image_id)

