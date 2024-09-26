import numpy as np
from mne.decoding import UnsupervisedSpatialFilter
from sklearn.decomposition import PCA, FastICA


def z_norm_by_trial(data):
    """
    Z-normalize data by trial, the input data is in the shape of (num_samples, num_channels, num_timesteps)
    @param data: data is in the shape of (num_samples, num_channels, num_timesteps)
    """
    norm_data = np.copy(data)
    for i in range(data.shape[0]):
        sample = data[i]
        mean = np.mean(sample, axis=(0, 1))
        std = np.std(sample, axis=(0, 1))
        sample_norm = (sample - mean) / std
        norm_data[i] = sample_norm
    return norm_data

def compute_pca_ica(X, n_components=20, pca=None, ica=None):
    """
    data will be normaly distributed after applying this dimensionality reduction
    @param X: input array
    @param n_components:
    @return:
    """
    if pca is None:
        pca = UnsupervisedSpatialFilter(PCA(n_components), average=False)
        pca_data = pca.fit_transform(X)
    else:
        pca_data = pca.transform(X)
    if ica is None:
        ica = UnsupervisedSpatialFilter(FastICA(n_components, whiten='unit-variance'), average=False)
        ica_data = ica.fit_transform(pca_data)
    else:
        ica_data = ica.transform(pca_data)

    return ica_data, pca, ica

def preprocess_samples_eeg_pupil(x_eeg, x_pupil, n_top_components=20, pca=None, ica=None, *args, **kwargs):
    x_eeg_znormed = z_norm_by_trial(x_eeg)
    x_pupil_znormed = z_norm_by_trial(x_pupil) if x_pupil is not None else None
    x_eeg_pca_ica, pca, ica = compute_pca_ica(x_eeg, n_top_components, pca=pca, ica=ica)
    return x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica


class Preprocessor:
    def __init__(self):
        """

        resample -> znorm -> pca -> ica

        znorm is always applied by default

        """
        self.pca = None
        self.ica = None

    def fit_transform(self, x, apply_pca_ica=False, *args, **kwargs):
        x = z_norm_by_trial(x)
        if apply_pca_ica:
            x, self.pca, self.ica = compute_pca_ica(x, *args, **kwargs)
        return x

    def transform(self, x):
        x = z_norm_by_trial(x)
        if self.pca is not None and self.ica is not None:
            x, *_ = compute_pca_ica(x)
        return x