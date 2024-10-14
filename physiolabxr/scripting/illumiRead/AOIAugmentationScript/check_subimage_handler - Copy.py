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


subimage_handler_path = 'data/subimage_handler.pkl'

with open(subimage_handler_path, 'rb') as f:
    subimage_handler = pickle.load(f)

print("subimage_handler loaded")