import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import pickle

from matplotlib import cm
from pylsl import pylsl


# from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import IMAGE_FORMAT


class ImageInfo():
    def __init__(self,
                 image_name = None,
                 original_image=None,
                 unrepaired_image = None,
                 sub_images=None,
                 label=None,
                 original_image_attention=None,
                 subimage_attention=None,
                 subimage_position=None,
                 ):
        self.image_name = image_name
        self.original_image = original_image
        self.unrepaired_image = unrepaired_image
        self.sub_images = sub_images
        self.label = label
        self.original_image_attention = original_image_attention
        self.subimage_attention = subimage_attention
        self.subimage_position = subimage_position


        self.image_on_screen_shape = None

    def get_sub_images_attention_rgba(self, normalized = False, alpha_threshold = 0.9, plot_results = False, colormap ='plasma'):

        sub_images_rgba = []

        for subimage_attention, subimage_position in zip(self.subimage_attention, self.subimage_position):

            subimage_attention_unpadded = subimage_attention[
                                          0:subimage_position[2][1]-subimage_position[0][1],
                                          0:subimage_position[2][0]-subimage_position[0][0]
                                          ]

            if normalized:
                subimage_attention_unpadded = subimage_attention_unpadded / subimage_attention_unpadded.max()

            sub_image_rgba = cm.plasma(subimage_attention_unpadded, alpha=alpha_threshold * subimage_attention_unpadded/np.max(subimage_attention_unpadded))
            if plot_results:
                plt.imshow(sub_image_rgba)
                plt.colorbar()
                plt.show()

            sub_images_rgba.append(sub_image_rgba)

        return sub_images_rgba

    def get_unreparied_rgba(self):
        return cv2.cvtColor(self.unrepaired_image, cv2.COLOR_BGR2RGBA)

    def update_perceptual_image_info(self, original_image_attention, subimage_attention, subimage_position):
        self.original_image_attention = original_image_attention
        self.subimage_attention = subimage_attention
        self.subimage_position = subimage_position

    # def get_original_image_attention(self, normalize_by_subimages = False):
    #
    #     # # if normalize_by_subimages:
    #     # #     # make a copy of the original image attention
    #     # #     original_image_attention = self.original_image_attention.copy()
    #     # #     for subimage_attention, subimage_position in zip(self.subimage_attention, self.subimage_position):
    #     # #
    #     # #         subimage_attention_unpadded = subimage_attention[
    #     # #                             0:subimage_position[2][1]-subimage_position[0][1],
    #     # #                             0:subimage_position[2][0]-subimage_position[0][0]
    #     # #                             ]
    #     # #
    #     # #         subimage_attention_unpadded = subimage_attention_unpadded / subimage_attention_unpadded.max()
    #     # #
    #     # #         original_image_attention[
    #     # #         subimage_position[0][0]:subimage_position[2][0],
    #     # #         subimage_position[0][1]:subimage_position[2][1]
    #     # #         ] = subimage_attention_unpadded
    #     # #
    #     # #
    #     # #     return original_image_attention
    #     # #
    #     # #
    #     # # else:
    #     #     return self.original_image_attention


def gray_image_to_rgba(image, normalize, alpha_threshold, uint8=False):

    if normalize:
        image = image / image.max()
    image_rgba= cm.plasma(image, alpha=alpha_threshold * image / np.max(image))

    if uint8:
        image_rgba = (image_rgba * 255).astype(np.uint8)

    return image_rgba









        # self.image_path = image_path
        # self.original_image = original_image
        # self.image_to_model = image_to_model
        # self.image_to_model_normalized = image_to_model_normalized
        # self.model_image_shape = model_image_shape
        # self.patch_shape = patch_shape
        # self.attention_grid_shape = attention_grid_shape
        # self.raw_attention_matrix = raw_attention_matrix
        # self.rollout_attention_matrix = rollout_attention_matrix
        # self.average_self_attention_matrix = average_self_attention_matrix
        # self.y_true = y_true
        # self.y_pred = y_pred



        # self.image_on_screen_shape = None
        # self.contours: ContourInfo


# class ContourInfo():
#     def __init__(self, contours, hierarchy):
#         self.contours = contours
#         self.hierarchy = hierarchy


def get_report_cleaned_image_info_dict(file_path, merge_dict=True):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    if merge_dict:
        data_dict_merged = {}
        for key in data_dict.keys():
            data_dict_merged.update(data_dict[key])
        return data_dict_merged
    else:
        return data_dict


def contours_to_lvt(contours, hierarchy, max_length=1024):
    '''
    Convert contours to lvt format.
    :param contours:
    :param max_length:
    :return:
    '''
    # data structure: [overflow_flag, contour_num, contour_0, contour_1, ...]
    # contour_i: [contour_index, hierarchy_info_length, hierarchy_info_0, hierarchy_info_1, ..., contour_vertices_num, contour_point_0_x, contour_point_0_y, contour_point_1_x, contour_point_1_y, ...]

    contours_lvt = [0]
    contours_lvt.append(len(contours))
    overflow_flag = False

    for contour_index, contour in enumerate(contours):
        contour_lvt = []
        contour_hierarchy = hierarchy[0][contour_index].tolist()
        hierarchy_info_length = len(contour_hierarchy)

        contour_vertices_num = len(contour)
        contour_points = contour.flatten().tolist()

        contour_lvt.append(contour_index)
        contour_lvt.append(hierarchy_info_length)
        contour_lvt += contour_hierarchy
        contour_lvt.append(contour_vertices_num)
        contour_lvt += contour_points

        contours_lvt += contour_lvt

        if len(contours_lvt) > max_length:
            overflow_flag = True
            break

    if not overflow_flag:
        # pad with zeros
        contours_lvt += [0] * (max_length - len(contours_lvt))
    else:
        contours_lvt[0] = 1

    return contours_lvt, overflow_flag


def sub_images_to_zmq_multipart(rgba_images, image_positions):

    item_list = []
    item_list.append(np.array(len(rgba_images)))
    for rgba_image, image_position in zip(rgba_images, image_positions):
        # convert from rgba to bgra
        image = (rgba_image*255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        # convert image to png format
        _, image_encoded = cv2.imencode('.png', image)
        image_encoded = image_encoded.tobytes()
        item_list.append(np.array(image_position))
        item_list.append(image_encoded)


    return item_list


def aoi_augmentation_zmq_multipart(topic, image_name, image_label, images_rgba):
    item_list = []
    item_list.append(bytes(topic, encoding='utf-8'))
    item_list.append(np.array(pylsl.local_clock()))

    item_list.append(bytes(image_name, encoding='utf-8'))
    item_list.append(bytes(image_label, encoding='utf-8'))

    for image in images_rgba:
        # convert from rgba to bgra
        # image = (image*255).astype(np.uint8)
        # assume the image is in range [0,255]
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        # convert image to png format
        _, image_encoded = cv2.imencode('.png', image)
        image_encoded = image_encoded.tobytes()
        item_list.append(image_encoded)

    return item_list

# def static_aoi_augmentation_to_zmq_multipart(topic, original_image, heatmap_overlay):
#     item_list = []
#     item_list.append(bytes(topic, encoding='utf-8'))
#     item_list.append(np.array(pylsl.local_clock()))







def get_image_on_screen_shape(original_image_width, original_image_height, image_width, image_height,
                              keep_aspect_ratio=True):
    '''
    Get the image shape on screen.
    :param original_image_width:
    :param original_image_height:
    :param image_width:
    :param image_height:
    :param keep_aspect_ratio:
    :return: int
    '''

    aspect_ratio = original_image_width / original_image_height

    if (image_width / image_height > aspect_ratio):
        image_width = image_height * aspect_ratio
    else:
        image_height = image_width / aspect_ratio

    return np.array([image_height, image_width]).astype(int)


def gaussian_filter(shape, center, sigma=1.0, normalized=True):
    """
    Create a Gaussian matrix with a given shape and center location.

    Parameters:
        shape (tuple): Shape of the matrix (rows, columns).
        center (tuple): Center location of the Gaussian (center_row, center_col).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        numpy.ndarray: The Gaussian matrix with the specified properties.
    """
    m, n = shape
    center_m, center_n = center

    # Create a grid of indices representing the row and column positions
    rows_indices, cols_indices = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')

    # Calculate the distance of each element from the center
    distances = np.sqrt((rows_indices - center_m) ** 2 + (cols_indices - center_n) ** 2)

    # Create the Gaussian matrix using the formula of the Gaussian distribution
    gaussian = np.exp(-distances ** 2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

    if normalized:
        gaussian_max, gaussian_min = gaussian.max(), gaussian.min()
        gaussian = (gaussian - gaussian_min) / (gaussian_max - gaussian_min)

    return gaussian


def image_coordinate_transformation(original_image_shape, target_image_shape, coordinate_on_original_image):
    """
    Transform the index on the original image to the target image using linear scaling.

    Parameters:
    - original_image_shape: Tuple (height, width) representing the shape of the original image.
    - target_image_shape: Tuple (height, width) representing the shape of the target image.
    - coordinate_on_original_image: Tuple (m, n) representing the coordinate on the original image.

    Returns:
    - Transformed coordinate on the target image as a tuple (m_transformed, n_transformed).
    """
    original_height, original_width = original_image_shape
    target_height, target_width = target_image_shape
    m, n = coordinate_on_original_image

    m_transformed = int(m * (target_height / original_height))
    n_transformed = int(n * (target_width / original_width))

    return np.array([m_transformed, n_transformed])


class GazeAttentionMatrixTorch():
    def __init__(self, image_shape=np.array([1000, 2000]),
                 attention_patch_shape=np.array([20, 20]),
                 sigma=20,
                 device=None):

        super().__init__()
        # self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_patch_shape = attention_patch_shape
        self.sigma = sigma
        self.device = device
        self.attention_grid_shape = np.array(
            [int(image_shape[0] / attention_patch_shape[0]), int(image_shape[1] / attention_patch_shape[1])])
        # self.attention_clutter_ratio = attention_clutter_ratio

        self._image_attention_buffer = torch.tensor(np.zeros(shape=self.image_shape), device=self.device)
        self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = self.image_shape - 1
        self._filter_map = torch.tensor(
            gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=self.sigma,
                            normalized=True), device=device)

        self._attention_patch_average_kernel = torch.tensor(
            np.ones(shape=attention_patch_shape) / (attention_patch_shape[0] * attention_patch_shape[1]), device=device)

        # # clutter removal
        # self._attention_grid_clutter_removal = ClutterRemoval(signal_clutter_ratio=0.1)
        # self._attention_grid_clutter_removal.evoke_data_processor()

        self._gaze_attention_grid_map = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)

    def get_image_attention_buffer(self, attention_center_location):

        x_offset_min = self._filter_map_center_location[0] - attention_center_location[0]
        x_offset_max = x_offset_min + self.image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - attention_center_location[1]
        y_offset_max = y_offset_min + self.image_shape[1]

        self._image_attention_buffer = self._filter_map[x_offset_min: x_offset_max,
                                       y_offset_min:y_offset_max].clone()  # this is a copy!!!

    def convolve_attention_grid_buffer(self):
        # pass
        # print(self._image_attention_buffer.shape)
        # if self._image_attention_buffer.shape[0] == 0 or self._image_attention_buffer.shape[1] == 0:
        #     print("GGGG")
        # self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)
        self._attention_grid_buffer = F.conv2d(
            input=self._image_attention_buffer.view(1, 1, self._image_attention_buffer.shape[0],
                                                    self._image_attention_buffer.shape[1]),
            weight=self._attention_patch_average_kernel.view(1, 1, self._attention_patch_average_kernel.shape[0],
                                                             self._attention_patch_average_kernel.shape[1]),
            stride=(self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1])).view(
            (self.attention_grid_shape[0], self.attention_grid_shape[1]))

    # def return_attention_grid(self):
    #     return self._attention_grid_clutter_removal.process_sample(self._attention_grid_buffer)
    def gaze_attention_grid_map_clutter_removal(self, attention_clutter_ratio=0.1):
        self._gaze_attention_grid_map = attention_clutter_ratio * self._gaze_attention_grid_map + (
                1 - attention_clutter_ratio) * self._attention_grid_buffer

    def get_gaze_attention_grid_map(self, flatten=True):
        gaze_attention_grid_map = self._gaze_attention_grid_map.cpu().numpy()
        if flatten:
            return gaze_attention_grid_map.flatten()
        else:
            return gaze_attention_grid_map

    # def get_attention_grid(self, flatten=True):
    #     attention_grid = self._attention_grid_buffer.cpu().numpy()
    #     if flatten:
    #         return attention_grid.flatten()
    #     else:
    #         return attention_grid

    def reset_image_attention_buffer(self):
        self._image_attention_buffer *= 0

    def reset_attention_grid_buffer(self):
        self._attention_grid_buffer *= 0

    def get_attention_grid_buffer(self):
        return self._attention_grid_buffer

    def plot_image_attention_buffer(self):
        plt.imshow(self._image_attention_buffer.cpu().numpy())
        plt.show()

    def plot_attention_grid_buffer(self):
        plt.imshow(self._attention_grid_buffer.cpu().numpy())
        plt.show()

    def plot_gaze_attention_grid_map(self):
        plt.imshow(self._gaze_attention_grid_map.cpu().numpy())
        plt.show()

    def plot_filter_map(self):
        plt.imshow(self._filter_map.cpu().numpy())
        plt.show()


class GazeAttentionMatrix():

    def __init__(self, device):

        self.device = device
        self.sigma = 60

        self.maximum_image_shape = None
        self._filter_size = None
        self._filter_map_center_location = None
        self.filter_map = None

        # self._attention_patch_average_kernel = None
        # self.attention_patch_shape = None
        # self.attention_grid_shape = None
        self.gaze_attention_pixel_map_buffer = None

    def set_maximum_image_shape(self, image_shape):
        self.maximum_image_shape = image_shape

        self._filter_size = self.maximum_image_shape * 2 - 1
        self._filter_map_center_location = self.maximum_image_shape - 1

        self._filter_map = torch.tensor(
            gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=self.sigma,
                            normalized=True), device=self.device)

    def setup_gaze_attention_matrix(self, image_shape):
        self.gaze_attention_pixel_map_buffer = torch.tensor(np.zeros(shape=image_shape),  device=self.device)

    # def set_attention_patch_shape(self, attention_patch_shape):
    #     self.attention_patch_shape = attention_patch_shape
    #     self._attention_patch_average_kernel = torch.tensor(
    #         np.ones(shape=self.attention_patch_shape) /
    #         (self.attention_patch_shape[0] * self.attention_patch_shape[1]), device=self.device)
    #
    #     self.attention_grid_shape = np.array([int(self.image_shape[0] / self.attention_patch_shape[0]),
    #                                           int(self.image_shape[1] / self.attention_patch_shape[1])])
    #     self.gaze_attention_grid_map_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape),
    #                                                        device=self.device)

    def get_gaze_on_image_attention_map(self, attention_center_location, image_shape):

        x_offset_min = self._filter_map_center_location[0] - attention_center_location[0]
        x_offset_max = x_offset_min + image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - attention_center_location[1]
        y_offset_max = y_offset_min + image_shape[1]

        gaze_on_image_attention_map = self._filter_map[x_offset_min: x_offset_max,
                                      y_offset_min:y_offset_max].clone()  # this is a copy!!!

        return gaze_on_image_attention_map

    # def get_patch_attention_map(self, gaze_on_image_attention_map):
    #     gaze_on_grid_attention_map = F.conv2d(
    #         input=gaze_on_image_attention_map.view(1, 1, gaze_on_image_attention_map.shape[0],
    #                                                gaze_on_image_attention_map.shape[1]),
    #         weight=self._attention_patch_average_kernel.view(1, 1, self._attention_patch_average_kernel.shape[0],
    #                                                          self._attention_patch_average_kernel.shape[1]),
    #         stride=(self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1])).view(
    #         (self.attention_grid_shape[0], self.attention_grid_shape[1]))
    #
    #     return gaze_on_grid_attention_map
    #
    def gaze_attention_pixel_map_clutter_removal(self, gaze_on_grid_attention_map, attention_clutter_ratio=0.1):
        if attention_clutter_ratio is None:
            self.gaze_attention_pixel_map_buffer = self.gaze_attention_pixel_map_buffer + gaze_on_grid_attention_map
        else:
            self.gaze_attention_pixel_map_buffer = attention_clutter_ratio * self.gaze_attention_pixel_map_buffer + (
                    1 - attention_clutter_ratio) * gaze_on_grid_attention_map

    #
    # def get_gaze_attention_grid_map(self, flatten=True):
    #     gaze_attention_grid_map = self.gaze_attention_grid_map_buffer.cpu().numpy()
    #     if flatten:
    #         return gaze_attention_grid_map.flatten()
    #     else:
    #         return gaze_attention_grid_map
    #
    #     # def accumulate_gaze_attention_grid(self, attention_center_location):
    #     #     gaze_on_image_attention_map = self.get_gaze_on_image_attention_map(attention_center_location)
    #     #     gaze_on_grid_attention_map = self.get_patch_attention_map(gaze_on_image_attention_map)
    #
    #     pass


class ViTAttentionMatrix():
    def __init__(self, attention_matrix=None):
        self.attention_matrix = attention_matrix
        self.patch_average_attention = None
        if self.attention_matrix:
            self.calculate_patch_average_attention_vector()

    def set_attention_matrix(self, attention_matrix):
        self.attention_matrix = attention_matrix

    def get_attention_matrix(self):
        return self.attention_matrix

    def calculate_patch_average_attention_vector(self):
        self.patch_average_attention = np.mean(self.attention_matrix, axis=1)

    def threshold_patch_average_attention(self, threshold=0.5):
        return np.where(self.patch_average_attention > threshold, 1, 0)

    def generate_random_attention_matrix(self, patch_num):
        self.attention_matrix = np.random.rand(patch_num, patch_num)


def generate_image_binary_mask(image, depth_first=False):
    if depth_first:
        image = np.moveaxis(image, 0, -1)

    # Convert the RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 254, 1, cv2.THRESH_BINARY_INV)
    return binary_mask


def generate_attention_grid_mask(image_mask, attention_patch_shape):
    kernel = torch.tensor(np.ones(shape=(attention_patch_shape[0], attention_patch_shape[1])), dtype=torch.float32)
    image_mask = torch.tensor(image_mask, dtype=torch.float32)

    attention_grid_mask = F.conv2d(input=image_mask.view(1, 1, image_mask.shape[0], image_mask.shape[1]),
                                   weight=kernel.view(1, 1, attention_patch_shape[0], attention_patch_shape[1]),
                                   stride=(attention_patch_shape[0], attention_patch_shape[1]))

    attention_grid_mask = attention_grid_mask.squeeze().cpu().numpy()
    attention_grid_mask = np.where(attention_grid_mask > 0, 1, 0)
    return attention_grid_mask


def attention_grid_mask_to_attention_matrix_mask(attention_grid_mask):
    attention_grid_mask_flatten = attention_grid_mask.flatten()
    attention_matrix_mask = np.ones(shape=(attention_grid_mask_flatten.shape[0], attention_grid_mask_flatten.shape[0]))

    for i in range(attention_grid_mask_flatten.shape[0]):
        if attention_grid_mask_flatten[i] == 0:
            attention_matrix_mask[i, :] = 0
            attention_matrix_mask[:, i] = 0

    return attention_matrix_mask

# def get_attention_matrix(image_path, image_shape, attention_patch_shape, mask_white=True):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (image_shape[1], image_shape[0]))
#     attention_grid_shape = (image_shape[0] // attention_patch_shape[0], image_shape[1] // attention_patch_shape[1])
#     attention_matrix = np.random.rand(attention_grid_shape[0]*attention_grid_shape[1], attention_grid_shape[0]*attention_grid_shape[1])
#
#     if mask_white:
#         binary_mask = generate_image_binary_mask(image)
#         attention_grid_mask = generate_attention_grid_mask(binary_mask, attention_patch_shape=attention_patch_shape)
#         attention_matrix_mask = attention_grid_mask_to_attention_matrix_mask(attention_grid_mask)
#         attention_matrix = attention_matrix * attention_matrix_mask
#
#     return attention_matrix
