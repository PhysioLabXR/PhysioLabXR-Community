from typing import Callable

import numpy as np
from numpy import ndarray


def get_indices_when(event_markers: ndarray, condition: Callable):
    """
    @param event_markers: 2D ndarray of event markers, the first dimension is the channels, the second is the time
    @param condition: a function that takes in an event marker and returns a boolean
    @return: ndarray of indices where the condition is true, if no indices are found, return None
    """
    found = np.argwhere(condition(event_markers))
    if len(found) == 0:
        return None
    else:
        return found[:, 1]