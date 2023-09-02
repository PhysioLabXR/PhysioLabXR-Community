import matplotlib.pyplot as plt
import numpy as np
import torch
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







class GazeAttentionMatrixTorch():
    def __init__(self, image_shape=np.array([1000, 2000]),
                 attention_patch_shape = np.array([20,20]),
                 sigma=20,
                 device=None):

        super().__init__()
        # self.attention_matrix = attention_matrix
        self.image_shape = image_shape
        self.attention_patch_shape = attention_patch_shape
        self.sigma = sigma
        self.device = device
        self.attention_grid_shape = np.array([int(image_shape[0]/attention_patch_shape[0]), int(image_shape[1]/attention_patch_shape[1])])
        # self.attention_clutter_ratio = attention_clutter_ratio

        self._image_attention_buffer = torch.tensor(np.zeros(shape=self.image_shape), device=self.device)
        self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)

        self._filter_size = self.image_shape * 2 - 1
        self._filter_map_center_location = self.image_shape - 1
        self._filter_map = torch.tensor(gaussian_filter(shape=self._filter_size, center=self._filter_map_center_location, sigma=self.sigma,
                                           normalized=True), device=device)

        self._attention_patch_average_kernel = torch.tensor(np.ones(shape=attention_patch_shape)/(attention_patch_shape[0] * attention_patch_shape[1]), device=device)

        # # clutter removal
        # self._attention_grid_clutter_removal = ClutterRemoval(signal_clutter_ratio=0.1)
        # self._attention_grid_clutter_removal.evoke_data_processor()

        self._gaze_attention_grid_map = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)




    def get_image_attention_buffer(self, attention_center_location):

            x_offset_min = self._filter_map_center_location[0] - attention_center_location[0]
            x_offset_max = x_offset_min + self.image_shape[0]

            y_offset_min = self._filter_map_center_location[1] - attention_center_location[1]
            y_offset_max = y_offset_min + self.image_shape[1]

            self._image_attention_buffer = self._filter_map[x_offset_min: x_offset_max, y_offset_min:y_offset_max].clone() # this is a copy!!!


    def convolve_attention_grid_buffer(self):
        # pass
        # print(self._image_attention_buffer.shape)
        # if self._image_attention_buffer.shape[0] == 0 or self._image_attention_buffer.shape[1] == 0:
        #     print("GGGG")
            # self._attention_grid_buffer = torch.tensor(np.zeros(shape=self.attention_grid_shape), device=self.device)
        self._attention_grid_buffer = F.conv2d(
            input=self._image_attention_buffer.view(1,1,self._image_attention_buffer.shape[0],self._image_attention_buffer.shape[1]),
            weight=self._attention_patch_average_kernel.view(1,1, self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1]),
            stride=(self._attention_patch_average_kernel.shape[0], self._attention_patch_average_kernel.shape[1])).view((self.attention_grid_shape[0], self.attention_grid_shape[1]))

    # def return_attention_grid(self):
    #     return self._attention_grid_clutter_removal.process_sample(self._attention_grid_buffer)
    def gaze_attention_grid_map_clutter_removal(self, attention_clutter_ratio=0.1):
        self._gaze_attention_grid_map = attention_clutter_ratio * self._gaze_attention_grid_map + (1 - attention_clutter_ratio) * self._attention_grid_buffer

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


