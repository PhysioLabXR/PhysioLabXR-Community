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
import pickle

class ImageInfo():
    def __init__(self,image_path, image, image_normalized, attention_patch_shape, attention_matrix, y=None, y_pred=None):
        self.image_path = image_path
        self.image = image
        self.image_normalized = image_normalized
        self.attention_patch_shape = attention_patch_shape
        self.attention_matrix = attention_matrix
        self.y = y
        self.y_pred = y_pred






##########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                   model_param='num-patch-32_image-size-1024-512')
##########################################################################


root_dir = r'D:\HaowenWei\UnityProject\PerceptualAOIAugmentation\Assets\Prefabs\OCTReportImages\Practice'
image_names = [file for file in os.listdir(root_dir) if file.endswith('.png')]


data_dict = {}

y_true = 'G'

for index, image_name in enumerate(image_names):

    image_path = os.path.join(root_dir, image_name)

    # get the prediction and attention matrix
    image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std) # the normalized image is z normalization
    y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device),
                                     collapse_attention_matrix=False)

    predicted_label = np.array([torch.argmax(y_pred).item()])
    print(f'y_pred: {y_pred}')
    decoded_label = compound_label_encoder.decode(predicted_label)
    print(f'Predicted label: {decoded_label}')

    plt.imshow(image.astype(np.uint8))
    plt.title(f'y_true: {[y_true]}, y_pred: {decoded_label}')
    plt.show()

    attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()

    class_token_attention = attention_matrix[0, 1:]

    attention_grid = class_token_attention.reshape(AOIAugmentationConfig.attention_grid_shape)

    attention_grid_upsample = np.repeat(attention_grid,2, axis=1)
    plt.imshow(attention_grid_upsample)
    plt.show()




    image_attention_info = ImageInfo(
        image_path=image_path,
        image=image,
        image_normalized=image_normalized,
        attention_patch_shape=AOIAugmentationConfig.attention_patch_shape,
        attention_matrix=attention_matrix,
        y=y_true,
        y_pred=decoded_label
    )


with open('../data/experiment_image_info/practice', 'wb') as file:
    # A new file will be created
    pickle.dump(data_dict, file)
