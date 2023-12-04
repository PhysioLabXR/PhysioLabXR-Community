import numpy as np
from scipy.interpolate import interp1d


def interpolate_nan(x):
    not_nan = np.logical_not(np.isnan(x))
    if np.sum(np.logical_not(not_nan)) / len(x) > 0.5:  # if more than half are nan
        raise ValueError("More than half of the given data array is nan")
    indices = np.arange(len(x))
    interp = interp1d(indices[not_nan], x[not_nan], fill_value="extrapolate")
    return interp(indices)


def interpolate_zeros(x):
    copy = np.copy(x)
    copy[copy == 0] = np.nan
    return interpolate_nan(copy)


def interpolate_array_nan(data_array):
    """
    :param data_array: channel first, time last
    """
    return np.array([interpolate_nan(x) for x in data_array])


def interpolate_epochs_nan(epoch_array):
    """
    :param data_array: channel first, time last
    """
    rtn = []
    rejected_count = 0
    for e in epoch_array:
        temp = []
        try:
            for x in e:
                temp.append(interpolate_nan(x))
        except ValueError:  # something wrong with this epoch, maybe more than half are nan
            rejected_count += 1
            continue  # reject this epoch
        rtn.append(temp)
    print("Rejected {0} epochs of {1} total".format(rejected_count, len(epoch_array)))
    return np.array(rtn)


def interpolate_epoch_zeros(e):
    copy = np.copy(e)
    copy[copy == 0] = np.nan
    return interpolate_epochs_nan(copy)

