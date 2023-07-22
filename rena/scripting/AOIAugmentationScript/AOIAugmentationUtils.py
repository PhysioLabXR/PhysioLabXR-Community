import time

import numpy as np

from enum import Enum
import matplotlib.pyplot as plt

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

def find_mapping(larger_shape = (1000, 2000), smaller_shape=(50, 25)):
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

def gaussian_filter(shape, center, sigma=1.0, normalized = True):
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
    distances = np.sqrt((rows_indices - center_m)**2 + (cols_indices - center_n)**2)

    # Create the Gaussian matrix using the formula of the Gaussian distribution
    gaussian = np.exp(-distances**2 / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    if normalized:
        gaussian_max, gaussian_min = gaussian.max(), gaussian.min()
        gaussian = (gaussian - gaussian_min) / (gaussian_max - gaussian_min)

    return gaussian

def extract_center_gaussian(filter_map, filter_map_center_location, image_shape, image_center_location):
    x_offset_min = filter_map_center_location[0] - image_center_location[0]
    x_offset_max = x_offset_min + image_shape[0]

    y_offset_min = filter_map_center_location[1] - image_center_location[1]
    y_offset_max = y_offset_min + image_shape[1]

    return filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]

class AOIAttentionMatrix:
    def __init__(self, attention_matrix, image_shape=np.array([1000, 2000]), attention_grid_shape=np.array([50, 25])):
        self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_grid_shape = attention_grid_shape

        self._image_attention_buffer = np.zeros(shape=self.image_shape)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = image_shape-1
        self._filter_map = gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=100, normalized = True)


    def get_center_gaussian(self):
        return extract_center_gaussian(self._filter_map, self._filter_map_center_location, self.image_shape, np.array([100,100]))








if __name__ == '__main__':
    a = AOIAttentionMatrix(attention_matrix=None)
    while 1:
        start_time = time.perf_counter_ns()
        b = a.get_center_gaussian()
        print(1e-6*(time.perf_counter_ns() - start_time))



