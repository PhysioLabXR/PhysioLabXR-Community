from typing import Callable

import numpy as np
from numpy import ndarray


def get_indices_when(event_markers: ndarray, condition: Callable, return_data: bool = False):
    """
    @param event_markers: 1D or 2D ndarray of event markers,
        if two dimensional, the first dimension is the channels, the second is the time
        if one dimensional, the time is assumed to be the only dimension
    @param condition: a function that takes in an event marker and returns a boolean
    @return: ndarray of indices where the condition is true, if no indices are found, return None
    """
    found = np.argwhere(condition(event_markers))
    if len(found) == 0:
        if return_data:
            return None, None
        else:
            return None

    if event_markers.ndim == 2:
        rtn_indices = found[:, 1]
    else:
        rtn_indices = found[:, 0]
    if return_data:
        return rtn_indices, event_markers[found]
    else:
        return rtn_indices