import time
from collections import deque
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, \
    GazeFilterFixationDetectionIVT, \
    tobii_gaze_on_display_area_to_image_matrix_index, GazeType, gaze_point_on_image_valid
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo
import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from eidl.utils.model_utils import get_trained_model, load_image_preprocess
import cv2


class ImageAttentionInfo():
    def __init__(self):
        self.image_path = None
        self.image = None
        self.image_shape = None
        self.attention_patch_shape = None
        self.attention_matrix = None



##########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                   model_param='num-patch-32_image-size-1024-512')
##########################################################################


root_dir = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\images'
image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith('.png')]


for image_path in image_paths:

    image_attention_info = ImageAttentionInfo()
    # get the prediction and attention matrix
    image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std) # the normalized image is z normalization
    y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device),
                                     collapse_attention_matrix=False)
    predicted_label = np.array([torch.argmax(y_pred).item()])
    decoded_label = compound_label_encoder.decode(predicted_label)
    print(f'Predicted label: {decoded_label}')

    # detach the attention matrix
    attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()

    # remove the class token
    attention_matrix = attention_matrix[1:, 1:]


    image_attention_info.image_path = image_path
    image_attention_info.image = image
    image_attention_info.image_shape = image.shape
    image_attention_info.attention_patch_shape = attention_matrix.shape
    image_attention_info.attention_matrix = attention_matrix





