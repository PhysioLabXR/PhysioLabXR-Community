import numpy as np
from imblearn.over_sampling import SMOTE
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

def rebalance_classes(x, y, by_channel=False, random_seed=None):
    """
    Resamples the data to balance the classes using SMOTE algorithm.

    Parameters:
        x (np.ndarray): Input data array of shape (epochs, channels, samples).
        y (np.ndarray): Target labels array of shape (epochs,).
        by_channel (bool): If True, balance the classes separately for each channel. Otherwise,
            balance the classes for the whole input data.

    Returns:
        tuple: A tuple containing the resampled input data and target labels as numpy arrays.
    """
    epoch_shape = x.shape[1:]

    if by_channel:
        y_resample = None
        channel_data = []
        channel_num = epoch_shape[0]

        # Loop through each channel and balance the classes separately
        for channel_index in range(0, channel_num):
            sm = SMOTE(random_state=random_seed)
            x_channel = x[:, channel_index, :]
            x_channel, y_resample = sm.fit_resample(x_channel, y)
            channel_data.append(x_channel)

        # Expand dimensions for each channel array and concatenate along the channel axis
        channel_data = [np.expand_dims(x, axis=1) for x in channel_data]
        x = np.concatenate([x for x in channel_data], axis=1)
        y = y_resample

    else:
        # Reshape the input data to 2D array and balance the classes
        x = np.reshape(x, newshape=(len(x), -1))
        sm = SMOTE(random_state=random_seed)
        x, y = sm.fit_resample(x, y)

        # Reshape the input data back to its original shape
        x = np.reshape(x, newshape=(len(x),) + epoch_shape)

    return x, y