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

        self.y = None
        self.y_pred = None





##########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                   model_param='num-patch-32_image-size-1024-512')
##########################################################################


root_dir = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\reports_cleaned\G'
image_names = [file for file in os.listdir(root_dir) if file.endswith('.png')]


data_dict = {}

y_true = 'G'

for image_name in image_names:

    image_path = os.path.join(root_dir, image_name)

    image_meta_info = image_name.split('_')


    image_attention_info = ImageAttentionInfo()
    # get the prediction and attention matrix
    image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std) # the normalized image is z normalization
    y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device),
                                     collapse_attention_matrix=False)
    predicted_label = np.array([torch.argmax(y_pred).item()])
    print(f'y_pred: {y_pred}')
    decoded_label = compound_label_encoder.decode(predicted_label)
    print(f'Predicted label: {decoded_label}')

    ##################################
    # plot the image
    # if y_true != decoded_label:
    plt.imshow(image.astype(np.uint8))
    plt.title(f'y_true: {[y_true]}, y_pred: {decoded_label}')
    plt.show()



    # detach the attention matrix
    attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()

    # remove the class token
    attention_matrix = attention_matrix[1:, 1:]


    image_attention_info.image_path = image_path
    image_attention_info.image = image
    image_attention_info.image_shape = image.shape
    image_attention_info.attention_patch_shape = attention_matrix.shape
    image_attention_info.attention_matrix = attention_matrix

    image_attention_info.y = image_meta_info[1]
    image_attention_info.y_pred = predicted_label

    # data_dict[int(image_meta_info[0])] = image_attention_info




    # post analysis
    average_attention = np.mean(attention_matrix, axis=1)

    # reshape to attention grid
    average_attention_grid = average_attention.reshape(AOIAugmentationConfig.attention_grid_shape[0], AOIAugmentationConfig.attention_grid_shape[1])

    # normalize the attention grid
    average_attention_grid_normalized = (average_attention_grid - np.min(average_attention_grid)) / (
            np.max(average_attention_grid) - np.min(average_attention_grid))

    # upscale the attention grid to the image size
    attention_grid_upscale_y = image.shape[0] // AOIAugmentationConfig.attention_grid_shape[0]
    attention_grid_upscale_x = image.shape[1] // AOIAugmentationConfig.attention_grid_shape[1]



    attention_image = np.repeat(np.repeat(average_attention_grid_normalized, attention_grid_upscale_x, axis=1),
                            attention_grid_upscale_y, axis=0)

    # print(f'attention_image.shape: {attention_image.shape}')
    threshold_mask = np.where(attention_image > 0.85, 1, 0)
    plt.imshow(threshold_mask)
    plt.show()
    time.sleep(0.1)


# static attention is [0, 1:]



print(data_dict.keys())