from typing import List

import numpy as np
from numpy import ndarray
from torch.utils.data import Dataset


class FixationDataset(Dataset):
    """
    b is the number of samples/fixations
    srate_pupil: sampling rate of the eye tracker: 200
    t_pupil: t_max_pupil - t_min_pupil
    Attributes:
        fixation_images: ndarray of shape (b, 3, h, w): uint8
        fixation_depth_images: ndarray of shape (b, 1, h, w): uint16
        pupil_epochs: ndarray of shape (b, 2, srate_pupil * t_pupil): float32
        eeg_epochs: ndarray of shape (b, c_eeg, srate_eeg * t_eeg): float32
        is_target: ndarray of shape (b): bool
    """

    def __init__(self):
        self.fixation_image: np.ndarray = np.empty()
        self.pupil_epochs: ndarray = np.empty()
        self.eeg_epochs: ndarray = np.empty()
        self.is_target: ndarray = np.empty()

    def extend(self, fixation_image: np.ndarray, fixation_depth_image: np.ndarray, pupil_epochs: ndarray,
               eeg_epochs: ndarray, is_target: ndarray):
        raise NotImplemented

    def __getitem__(self, index):
        return self.fixation_image[index], self.fixation_depth_image[index], self.pupil_epochs[index], self.eeg_epochs[
            index], self.is_target[index]
