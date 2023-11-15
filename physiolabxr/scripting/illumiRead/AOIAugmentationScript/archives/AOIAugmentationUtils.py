import time
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


class ExperimentState(Enum):
    CalibrationState = 1
    StartState = 2
    IntroductionInstructionState = 3
    PracticeInstructionState = 4
    NoAOIAugmentationInstructionState = 5
    NoAOIAugmentationState = 6
    StaticAOIAugmentationInstructionState = 7
    StaticAOIAugmentationState = 8
    InteractiveAOIAugmentationInstructionState = 9
    InteractiveAOIAugmentationState = 10
    FeedbackState = 11
    EndState = 12


class NetworkConfig(Enum):
    ZMQPortNumber = 6667


def generate_random_attention_matrix(grid_shape=(25, 50)):
    patch_num = grid_shape[0] * grid_shape[1]
    attention_matrix = np.random.random((patch_num, patch_num))
    return attention_matrix


def find_mapping(larger_shape=(1000, 2000), smaller_shape=(50, 25)):
    larger_width, larger_height = larger_shape
    smaller_width, smaller_height = smaller_shape

    mapping = [[None for _ in range(larger_width)] for _ in range(larger_height)]

    for y in range(larger_height):
        for x in range(larger_width):
            smaller_x = int(x * smaller_width / larger_width)
            smaller_y = int(y * smaller_height / larger_height)
            mapping[y][x] = (smaller_x, smaller_y)

    return mapping


def find_mapping_np(larger_shape, smaller_shape):
    larger_width, larger_height = larger_shape
    smaller_width, smaller_height = smaller_shape

    # Create arrays of x and y coordinates for the larger matrix
    x_larger = np.arange(larger_width)
    y_larger = np.arange(larger_height)

    # Calculate the corresponding x and y coordinates in the smaller matrix
    x_smaller = (x_larger * smaller_width / larger_width).astype(int)
    y_smaller = (y_larger * smaller_height / larger_height).astype(int)

    # Use meshgrid to create a 2D grid of the smaller matrix coordinates
    smaller_x_grid, smaller_y_grid = np.meshgrid(x_smaller, y_smaller)

    # Combine the x and y coordinates to get the final mapping
    mapping = np.dstack((smaller_x_grid, smaller_y_grid))

    return mapping


# def gaussian_filter(size, center, sigma):
#     x_c, y_c = center
#     x, y = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
#
#     distance_squared = (x - x_c) ** 2 + (y - y_c) ** 2
#     gaussian = np.exp(-distance_squared / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
#     return gaussian

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


def extract_center_gaussian(filter_map, filter_map_center_location, image_shape, image_attention_location):
    x_offset_min = filter_map_center_location[0] - image_attention_location[0]
    x_offset_max = x_offset_min + image_shape[0]

    y_offset_min = filter_map_center_location[1] - image_attention_location[1]
    y_offset_max = y_offset_min + image_shape[1]

    return filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]


def tobii_to_screen_canvas_coordinate(tobii_gaze_on_display_x, tobii_gaze_on_display_y, screen_width, screen_height):
    on_canvas_coordinate_x = screen_width * (tobii_gaze_on_display_x - 0.5)
    on_canvas_coordinate_y = screen_height * (0.5 - tobii_gaze_on_display_y)

    return on_canvas_coordinate_x, on_canvas_coordinate_y


def tobii_to_image_pixel_index(tobii_gaze_on_display_x, tobii_gaze_on_display_y,
                               screen_width, screen_height,
                               image_center_x, image_center_y,
                               image_width, image_height):
    on_canvas_coordinate_x, on_canvas_coordinate_y = tobii_to_screen_canvas_coordinate(tobii_gaze_on_display_x,
                                                                                       tobii_gaze_on_display_y,
                                                                                       screen_width,
                                                                                       screen_height)

    image_top_left_conor_x = image_center_x - image_width / 2
    image_top_left_conor_y = image_center_y + image_height / 2

    image_pixel_index_x = on_canvas_coordinate_x - image_top_left_conor_x
    image_pixe_index_y = -on_canvas_coordinate_y + image_top_left_conor_y

    return image_pixel_index_x, image_pixe_index_y

def screen_space_to_image_space(screen_space_location, image_shape, screen_shape):
    screen_width, screen_height = screen_shape
    image_width, image_height = image_shape

    image_x = int(screen_space_location[0] * image_width / screen_width)
    image_y = int(screen_space_location[1] * image_height / screen_height)

    return image_x, image_y


class AOIAttentionMatrix:
    def __init__(self, attention_matrix, image_shape=np.array([1000, 2000]), attention_grid_shape=np.array([25, 50]),
                 device=None):
        self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_grid_shape = attention_grid_shape

        self._image_attention_buffer = np.zeros(shape=self.image_shape)
        self._attention_grid_buffer = np.zeros(shape=self.attention_grid_shape)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = image_shape - 1
        self._filter_map = gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=100,
                                           normalized=True)

    # def get_center_gaussian(self):
    #     return extract_center_gaussian(self._filter_map, self._filter_map_center_location, self.image_shape,
    #                                    np.array([100, 100]))

    # def add_attention(self):

    def add_attention(self, image_center_location):
        x_offset_min = self._filter_map_center_location[0] - image_center_location[0]
        x_offset_max = x_offset_min + self.image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - image_center_location[1]
        y_offset_max = y_offset_min + self.image_shape[1]

        self._image_attention_buffer += self._filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]

    def decay(self):
        # gaussian decay
        self._image_attention_buffer = self._image_attention_buffer / 2

    def reset_image_attention_buffer(self):
        self._image_attention_buffer *= 0


class AOIAttentionMatrixTensor:
    def __init__(self, attention_matrix, image_shape=np.array([1000, 2000]), attention_grid_shape=np.array([25, 50]),
                 device=None):
        self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_grid_shape = attention_grid_shape
        self.device = device
        self._image_attention_buffer = torch.tensor(np.zeros(shape=self.image_shape), device=self.device)
        self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=device)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = image_shape - 1
        self._filter_map = torch.tensor(
            gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=100,
                            normalized=True), device=device)

    def add_attention(self, image_center_location):
        x_offset_min = self._filter_map_center_location[0] - image_center_location[0]
        x_offset_max = x_offset_min + self.image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - image_center_location[1]
        y_offset_max = y_offset_min + self.image_shape[1]

        self._image_attention_buffer += self._filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]

    def decay(self):
        # gaussian decay
        self._image_attention_buffer = self._image_attention_buffer / 2

    def reset_image_attention_buffer(self):
        self._image_attention_buffer *= 0



if __name__ == '__main__':
    device = torch.device('cuda:0')

    image_shape = np.array([1000, 2000])
    attention_grid_shape = np.array([25, 50])
    a = AOIAttentionMatrixTensor(attention_matrix=None, image_shape=image_shape,
                                 attention_grid_shape=attention_grid_shape, device=device)
    a.add_attention(image_center_location=[100, 100])
    a.decay()

    while 1:
        attention_add_start = time.perf_counter_ns()
        a.add_attention(image_center_location=[100, 100])
        attention_add_time = time.perf_counter_ns() - attention_add_start

        attention_decay_start = time.perf_counter_ns()
        a.decay()
        attention_decay_time = time.perf_counter_ns() - attention_decay_start

        detach_start = time.perf_counter_ns()
        b = a._image_attention_buffer.cpu()
        detach_time = time.perf_counter_ns() - detach_start

        print(attention_add_time * 1e-6, attention_decay_time * 1e-6, detach_time * 1e-6)
