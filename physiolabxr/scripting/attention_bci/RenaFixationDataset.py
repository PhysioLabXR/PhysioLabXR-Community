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
        # feature: now uses the dummpy empty array
        self.fixation_image = np.empty((0,), dtype=np.uint8)
        self.fixation_depth_image = np.empty((0,), dtype=np.uint16)
        self.pupil_epochs = np.empty((0,), dtype=np.float32)
        self.eeg_epochs = np.empty((0,), dtype=np.float32)
        self.is_target = np.empty((0,), dtype=bool)

    def extend(
            self,
            fixation_image: np.ndarray,
            fixation_depth_image: np.ndarray,
            pupil_epochs: np.ndarray,
            eeg_epochs: np.ndarray,
            is_target: np.ndarray,
    ) -> None:
        """
        Append one *batch* of samples to the internal arrays.

        All inputs must have the *same* first-dimension length B (batch size).
        Subsequent calls can use any batch size; the arrays will be concatenated
        along axis 0.
        """
        # 1️. sanity – make sure the batch dimensions match
        B = fixation_image.shape[0]
        if not (
                fixation_depth_image.shape[0] == pupil_epochs.shape[0] ==
                eeg_epochs.shape[0] == is_target.shape[0] == B
        ):
            raise ValueError("All input arrays must share the same batch size")

        # 2️. append (concatenate) to each storage array
        def _append(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
            if dst.size == 0:  # first call → just take src
                return src.copy()
            return np.concatenate((dst, src), axis=0)

        self.fixation_image = _append(self.fixation_image, fixation_image)
        self.fixation_depth_image = _append(self.fixation_depth_image, fixation_depth_image)
        self.pupil_epochs = _append(self.pupil_epochs, pupil_epochs)
        self.eeg_epochs = _append(self.eeg_epochs, eeg_epochs)
        self.is_target = _append(self.is_target, is_target)

        # (optional) return new dataset length
        # return self.fixation_image.shape[0]

    def __getitem__(self, index):
        return self.fixation_image[index], self.fixation_depth_image[index], self.pupil_epochs[index], self.eeg_epochs[
            index], self.is_target[index]
