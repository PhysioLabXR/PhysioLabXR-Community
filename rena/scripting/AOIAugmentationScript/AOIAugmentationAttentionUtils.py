import time
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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




class AOIAttentionMatrixTorch(nn.Module):
    def __init__(self, attention_matrix, image_shape=np.array([1000, 2000]), attention_patch_shape = np.array([20,20]), #attention_grid_shape=np.array([25, 50]),
                 device=None):

        super().__init__()
        self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_patch_shape = attention_patch_shape
        self.device = device
        self.attention_grid_shape = np.array([int(image_shape[0]/attention_patch_shape[0]), int(image_shape[1]/attention_patch_shape[1])])


        self._image_attention_buffer = torch.tensor(np.zeros(shape=self.image_shape), device=self.device)
        self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = image_shape - 1
        self._filter_map = torch.tensor(gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=20,
                                           normalized=True), device=device)

        self._attention_patch_average_kernel = torch.tensor(np.ones(shape=attention_patch_shape)/(attention_patch_shape[0] * attention_patch_shape[1]), device=device)





    def add_attention(self, image_center_location):
        x_offset_min = self._filter_map_center_location[0] - image_center_location[0]
        x_offset_max = x_offset_min + self.image_shape[0]

        y_offset_min = self._filter_map_center_location[1] - image_center_location[1]
        y_offset_max = y_offset_min + self.image_shape[1]

        self._image_attention_buffer += self._filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max]

    def decay(self):
        # gaussian decay
        self._image_attention_buffer = self._image_attention_buffer / 2

    def attention_grid(self):
        # pass
        self._attention_grid_buffer = F.conv2d(
            input=self._image_attention_buffer.view(1,1,self._image_attention_buffer.shape[0],self._image_attention_buffer.shape[1]),
            weight=self._attention_patch_average_kernel.view(1,1, self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1]),
            stride=(self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1]))


    def reset_image_attention_buffer(self):
        self._image_attention_buffer *= 0

    @property
    def attention_grid_buffer(self):
        return self._attention_grid_buffer


if __name__ == '__main__':
    device = torch.device('cuda:0')

    image_shape = np.array([500, 1000])
    attention_grid_shape = np.array([25, 50])
    attention_patch_shape = np.array([20,20])
    a = AOIAttentionMatrixTorch(attention_matrix=None, image_shape=image_shape, attention_patch_shape=attention_patch_shape, device=device)
    a.add_attention(image_center_location=[100,100])
    a.decay()
    a.attention_grid()

    while 1:
        attention_add_start = time.perf_counter_ns()
        a.add_attention(image_center_location=[100, 100])
        attention_add_time = time.perf_counter_ns()-attention_add_start

        attention_decay_start = time.perf_counter_ns()
        a.decay()
        attention_decay_time = time.perf_counter_ns()-attention_decay_start


        attention_grid_average_start = time.perf_counter_ns()
        a.attention_grid()
        attention_grid_average_time = time.perf_counter_ns()-attention_grid_average_start

        detach_start = time.perf_counter_ns()
        b = a.attention_grid_buffer.view(25, 50).cpu()
        detach_time = time.perf_counter_ns() - detach_start

        print(attention_add_time*1e-6, attention_decay_time*1e-6, attention_grid_average_time*1e-6, detach_time*1e-6)
        print('time cost:', (attention_add_time+attention_decay_time+attention_grid_average_time+detach_time)*1e-6)

        # plt.imshow(a._attention_grid_buffer.view(25, 50).cpu())
        # plt.show()

        # plt.imshow(a._image_attention_buffer.view(500, 1000).cpu())
        # plt.show()