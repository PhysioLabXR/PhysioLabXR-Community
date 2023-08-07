import enum
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rena.scripting.AOIAugmentationScript.AOIAugmentationConfig import TobiiProFusionChannel


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




class GazeAttentionMatrixTorch():
    def __init__(self, image_shape=np.array([1000, 2000]), attention_patch_shape = np.array([20,20]), sigma=20, #attention_grid_shape=np.array([25, 50]),
                 device=None):

        super().__init__()
        # self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_patch_shape = attention_patch_shape
        self.device = device
        self.attention_grid_shape = np.array([int(image_shape[0]/attention_patch_shape[0]), int(image_shape[1]/attention_patch_shape[1])])


        self._image_attention_buffer = torch.tensor(np.zeros(shape=self.image_shape), device=self.device)
        self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = image_shape - 1
        self._filter_map = torch.tensor(gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=sigma,
                                           normalized=True), device=device)

        self._attention_patch_average_kernel = torch.tensor(np.ones(shape=attention_patch_shape)/(attention_patch_shape[0] * attention_patch_shape[1]), device=device)





    def add_attention(self, attention_center_location):
        x_offset_min = self._filter_map_center_location[0] - attention_center_location[0]
        x_offset_max = x_offset_min + self.image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - attention_center_location[1]
        y_offset_max = y_offset_min + self.image_shape[1]

        self._image_attention_buffer += self._filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]

    def min_max_normalize_attention(self):
        self._image_attention_buffer = (self._image_attention_buffer - self._image_attention_buffer.min()) / (self._image_attention_buffer.max() - self._image_attention_buffer.min())

    def decay(self, decay_factor=0.5):
        # gaussian decay
        self._image_attention_buffer = self._image_attention_buffer * decay_factor

    def calculate_attention_grid(self):
        # pass
        self._attention_grid_buffer = F.conv2d(
            input=self._image_attention_buffer.view(1,1,self._image_attention_buffer.shape[0],self._image_attention_buffer.shape[1]),
            weight=self._attention_patch_average_kernel.view(1,1, self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1]),
            stride=(self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1])).view((self.attention_grid_shape[0], self.attention_grid_shape[1]))

    def threshold_attention_grid_vector(self, flatten=True, threshold=0.5, dtype=np.float64):
        # attention_grid = self._attention_grid_buffer.cpu().numpy()
        attention_grid = np.where(self._attention_grid_buffer.cpu().numpy() > threshold, 1, 0).astype(dtype)
        if flatten:
            return attention_grid.flatten()
        else:
            return attention_grid

    # def threshold_attention_grid_vector_buffer(self, flatten=True, threshold=0.5):


    def get_attention_grid(self, flatten=True):
        attention_grid = self._attention_grid_buffer.cpu().numpy()
        if flatten:
            return attention_grid.flatten()
        else:
            return attention_grid


    def reset_image_attention_buffer(self):
        self._image_attention_buffer *= 0

    def get_attention_grid_buffer(self):
        return self._attention_grid_buffer

    def plot_image_attention_buffer(self):
        plt.imshow(self._image_attention_buffer.cpu().numpy())
        plt.show()

    def plot_attention_grid_buffer(self):
        plt.imshow(self._attention_grid_buffer.cpu().numpy())
        plt.show()


class ViTAttentionMatrix():
    def __init__(self, attention_matrix = None):
        self.attention_matrix = attention_matrix
        self.patch_average_attention = None
        if self.attention_matrix:
            self.calculate_patch_average_attention_vector()

    def set_attention_matrix(self, attention_matrix):
        self.attention_matrix = attention_matrix
        self.patch_average_attention()

    def get_attention_matrix(self):
        return self.attention_matrix

    def calculate_patch_average_attention_vector(self):
        self.patch_average_attention = np.mean(self.attention_matrix, axis=1)

    def threshold_patch_average_attention(self, threshold=0.5):
        return np.where(self.patch_average_attention > threshold, 1, 0)

    def generate_random_attention_matrix(self, patch_num):
        self.attention_matrix = np.random.rand(patch_num, patch_num)

# if __name__ == '__main__':
#     a  = ViTAttentionMatrix()
#     a.generate_random_attention_matrix(10)
#     a.calculate_patch_average_attention_vector()
#     print(a.attention_matrix)
#     b = a.threshold_patch_average_attention(0.5)

