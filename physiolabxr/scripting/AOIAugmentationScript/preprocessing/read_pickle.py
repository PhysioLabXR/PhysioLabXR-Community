import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageInfo():
    def __init__(self, image_path, image, image_normalized, model_image_shape,
                 patch_shape, attention_grid_shape,
                 raw_attention_matrix,
                 rollout_attention_matrix, average_self_attention_matrix,
                 y=None, y_pred=None):
        self.image_path = image_path
        self.image = image
        self.image_normalized = image_normalized
        self.model_image_shape = model_image_shape
        self.patch_shape = patch_shape
        self.attention_grid_shape = attention_grid_shape
        self.raw_attention_matrix = raw_attention_matrix
        self.rollout_attention_matrix = rollout_attention_matrix
        self.average_self_attention_matrix = average_self_attention_matrix
        self.y = y
        self.y_pred = y_pred




read_pickle_file = 'data_dict.pkl'

with open(read_pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

