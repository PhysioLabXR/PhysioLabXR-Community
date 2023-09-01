import numpy as np
from scipy.interpolate import interp1d


def interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
    if np.sum(np.logical_not(not_nan)) / len(x) > 0.5:  # if more than half are nan
        raise ValueError("More than half of the given data array is nan")
    indices = np.arange(len(x))
    interp = interp1d(indices[not_nan], x[not_nan], fill_value="extrapolate")
    return interp(indices)


def interpolate_array_nan(data_array):
    """
    :param data_array: channel first, time last
    """
    return np.array([interpolate_nan(x) for x in data_array])


def time_to_index(timestamps, time):
    return np.argmin(np.abs(timestamps - time))

def string_to_enum(enum_type, string_value):
    try:
        return enum_type[string_value]
    except KeyError:
        raise ValueError(f"'{string_value}' is not a valid value for {enum_type.__name__}")
