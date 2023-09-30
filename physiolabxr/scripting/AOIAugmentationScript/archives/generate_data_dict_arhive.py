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


# data dict structure:
# data_dict = {
#     'image_path': image_path,
#     'image_rgb': image_rgb, (0-1)
#     'y': label, (G or S)
#     'y_pred': prediction, (G or S)
#     'attention_matrix_raw': attention_matrix_raw, (0-1)
#     'attention_matrix': attention_matrix, (0-1)
# }

class ImageAttentionClass():
    def __init__(self, image_path, image, y, y_pred, patch_shape, patch_grid_shape, attention_matrix_raw):
        self.image_path = image_path
        self.image = image
        self.y = y
        self.y_pred = y_pred
        self.patch_shape = patch_shape
        self.patch_grid_shape = patch_grid_shape
        self.attention_matrix_raw = attention_matrix_raw


# replace the image path to yours

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                     model_param='num-patch-32_image-size-1024-512')

# note: we have to perform the global normalization on the image before feeding it to the model
image_path = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\test\edge_detector\images\02_8981_OS_2021_widefield_report.png'
image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)

# get the prediction
y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device),
                                 collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])
decoded_label = compound_label_encoder.decode(predicted_label)

print(f'Predicted label: {decoded_label}')

attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()
attention_matrix_without_class_token = attention_matrix[1:, 1:]

average_attention = np.mean(attention_matrix_without_class_token, axis=1)
average_attention_grid = average_attention.reshape(32, 32)
average_attention_grid_normalized = (average_attention_grid - np.min(average_attention_grid)) / (
            np.max(average_attention_grid) - np.min(average_attention_grid))

# Upsample factors
attention_grid_upscale_x = 1024 // 32
attention_grid_upscale_y = 512 // 32

# Upsample the image using NumPy's repeat function
attention_image = np.repeat(np.repeat(average_attention_grid_normalized, attention_grid_upscale_x, axis=1),
                            attention_grid_upscale_y, axis=0)

# minimax normalization
# attention_image = (attention_image - np.min(attention_image)) / (np.max(attention_image) - np.min(attention_image))

cmap = plt.get_cmap('hot')
attention_image_heatmap = cmap(attention_image)
plt.imshow(attention_image_heatmap)
plt.show()

th = np.where(attention_image > 0.85, 1, 0)
plt.imshow(th)
plt.show()
