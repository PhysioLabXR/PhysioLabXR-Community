from multiprocessing import Pool

import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, w0, bw, fs, channel_format='first'):
    assert len(data.shape) == 2

    quality_factor = w0 / bw
    b, a = iirnotch(w0, quality_factor, fs)

    if channel_format == 'last':
        output_signal = np.array([filtfilt(b, a, data[:, i]) for i in range(data.shape[-1])])
    elif channel_format == 'first':
        output_signal = np.array([filtfilt(b, a, data[i, :]) for i in range(data.shape[0])])
    else:
        raise Exception('Unrecognized channgel format, must be either "first" or "last"')
    return output_signal


def baseline_als(y, lam, p, niter):
    """
    base line correction based on Asymmetric Least Squares Smoothing
    ref: https://stackoverflow.com/questions/29156532/python-baseline-correction-library
    :rtype: object
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def baseline_correction(data, lam, p, niter=10, channel_format='first', njobs=20):
    pool = Pool(processes=njobs)
    if channel_format == 'last':
        pool_result = [pool.apply(baseline_als, args=(data[:, i], lam, p, niter)) for i in range(data.shape[-1])]
    elif channel_format == 'first':
        pool_result = [pool.apply(baseline_als, args=(data[i, :], lam, p, niter)) for i in range(data.shape[0])]
        # output_signal = np.array([baseline_als(data[i, :], lam, p, niter) for i in range(data.shape[0])])
    else:
        raise Exception('Unrecognized channgel format, must be either "first" or "last"')
    output_signal = np.array(pool_result)
    return output_signal
