import os

# analysis parameters ######################################################################################
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


def z_norm_hdca(x, _mean=None, _std=None):
    if _mean is None or _std is None:
        _mean = np.mean(x, axis=0, keepdims=True)
        _std = np.std(x, axis=0, keepdims=True)
    return (x - _mean) / _std

def compute_forward(x_windowed, y, projection):
    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_windowed.shape
    activation = np.empty((2, num_channels, num_windows, num_timepoints_per_window))
    for class_index in np.sort(np.unique(y)):  # for test set, in increasing order
        this_x = x_windowed[y == class_index]
        this_projection = projection[y == class_index]
        for j in range(num_windows):
            this_x_window = this_x[:, :, j, :].reshape(this_x.shape[0], -1).T
            # z_window = np.array([np.dot(weights_channelWindow[j], this_x[trial_index, :, j, :].reshape(-1)) for trial_index in range(this_x.shape[0])])
            # z_window = z_window.reshape((-1, 1)) # change to a col vector
            this_projection_window = this_projection[:, j]
            a = (np.matmul(this_x_window, this_projection_window) / np.matmul(this_projection_window.T, this_projection_window).item()).reshape((num_channels, num_timepoints_per_window))
            activation[class_index, :, j] = a
    return activation

def _train_compute_window_projections(x_train_windowed, x_test_windowed, y_train):
    weights_channelWindow, projection_train_window_trial, ldas = _train_window_lda(x_train_windowed, y_train)
    projectionTest_window_trial = _compute_window_lda_projections(x_test_windowed, ldas)
    return weights_channelWindow, projection_train_window_trial, projectionTest_window_trial, ldas


def _compute_window_lda_projections(x_windowed, ldas):
    num_trials, num_channels, num_windows, num_timepoints_per_window = x_windowed.shape
    projections = np.empty((num_trials, num_windows))
    for k in range(num_windows):  # iterate over different windows
        this_x = x_windowed[:, :, k, :].reshape((num_trials, -1))
        projections[:, k] = ldas[k].transform(this_x).squeeze(axis=1)
    return projections

def _train_window_lda(x_train_windowed, y_train):
    num_train_trials, num_channels, num_windows, num_timepoints_per_window = x_train_windowed.shape

    weights_channelWindow = np.empty((num_windows, num_channels * num_timepoints_per_window))
    projection_train_window_trial = np.empty((num_train_trials, num_windows))

    ldas = []
    for k in range(num_windows):  # iterate over different windows
        this_x_train = x_train_windowed[:, :, k, :].reshape((num_train_trials, -1))
        lda = LinearDiscriminantAnalysis(solver='svd')
        projection_train_window_trial[:, k] = lda.fit_transform(this_x_train, y_train).squeeze(axis=1)
        ldas.append(lda)
        _weights = np.squeeze(lda.coef_, axis=0)
        weights_channelWindow[k] = _weights
    return weights_channelWindow, projection_train_window_trial, ldas

def eval_crossbin_model(x_project, y, model):
    y_pred = model.predict(x_project)
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    return y_pred, roc_auc, fpr, tpr

def solve_crossbin_weights(projection_train, projection_test, y_train, y_test, num_windows, verbose=0, max_iter=5000, random_seed=None):
    model = LogisticRegression( max_iter=max_iter, fit_intercept=True, penalty='l2', solver='saga').fit(projection_train, y_train)
    y_pred = model.predict(projection_test)
    cross_window_weights = np.squeeze(model.coef_, axis=0)

    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='example estimator')
    # fig = plt.figure(figsize=(10, 10), constrained_layout=True)
    # display.plot(ax=plt.gca())
    # plt.tight_layout()
    # plt.show()

    # plt.plot(cross_window_weights)
    # plt.xticks(ticks=list(range(1, num_windows + 1)), labels=[str(x) for x in list(range(1, num_windows + 1))])
    # plt.xlabel("100ms windowed bins")
    # plt.ylabel("Cross-bin weights")
    # plt.tight_layout()
    # plt.show()
    return cross_window_weights, roc_auc, fpr, tpr, model



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

def z_norm_projection(x_train, x_test):
    assert len(x_train.shape) == len(x_test.shape) == 2
    projection_mean = np.mean(np.concatenate((x_train, x_test), axis=0), axis=0, keepdims=True)
    projection_std = np.std(np.concatenate((x_train, x_test), axis=0), axis=0, keepdims=True)

    return (x_train - projection_mean) / projection_std, (x_test - projection_mean) / projection_std, projection_mean, projection_std


def plot_forward(activation, event_names, split_window, num_windows, exg_srate, eeg_montage, notes):
    import mne
    from mne.viz import plot_topomap
    eeg_channel_names = eeg_montage.ch_names
    info = mne.create_info(
        eeg_channel_names,
        sfreq=exg_srate,
        ch_types=['eeg'] * len(eeg_channel_names))
    info.set_montage(eeg_montage)

    fig = plt.figure(figsize=(22, 10), constrained_layout=True)
    subfigs = fig.subfigures(2, 1)
    # fig, axs = plt.subplots(2, num_windows - 1, figsize=(22, 10), sharey=True)  # sharing vmax and vmin
    for class_index, e_name in enumerate(event_names):
        axes = subfigs[class_index].subplots(1, num_windows, sharey=True)
        for i in range(num_windows):
            a = np.mean(activation[class_index, :, i, :], axis=1)
            plot_topomap(a, info, axes=axes[i - 1], show=False, res=512, vlim=(np.min(activation), np.max(activation)))
            axes[i - 1].set_title(f"{int((i - 1) * split_window * 1e3)}-{int(i * split_window * 1e3)}ms")
        subfigs[class_index].suptitle(e_name)
    fig.suptitle(f"{notes} Activation map from Fisher Discriminant Analysis. ", fontsize='x-large')
    plt.show()


class HDCA():
    def __init__(self, event_names):
        """
        :param exg_srate: sampling rate of exg data
        :param verbose: whether to print out information
        """
        self._encoder = None

        self.exg_srate = None
        self.split_window_eeg = None
        self.split_window_pupil = None
        self.crossbin_model_combined = None
        self.crossbin_model_eeg = None
        self.crossbin_model_pupil = None

        self.eeg_mean = None
        self.eeg_std = None
        self.pupil_mean = None
        self.pupil_std = None

        self.window_pupil_ldas = None
        self.window_eeg_ldas = None

        self.num_channels_pupil = None
        self.num_windows_pupil = None
        self.num_timepoints_per_window_pupil = None
        self.num_windows_eeg = None
        self.num_channels_eeg = None
        self.num_timepoints_per_window_eeg = None
        self.event_names = event_names
        self.split_size_eeg = None
        self.split_size_pupil = None
        self.num_eeg_windows = None
        self.num_pupil_windows = None

    def fit(self, x_eeg, x_eeg_pca_ica, x_pupil, y, is_plots=False, notes="", num_folds=10, exg_srate=200, split_window_eeg=100e-3, split_window_pupil=500e-3, eyetracking_srate=20, random_seed=None, verbose=0, eeg_montage=None, *args, **kwargs):
        self._use_pupil = x_pupil is not None
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        self._encoder = lambda y: label_encoder.transform(y)
        y = self._encoder(np.copy(y))

        if self._use_pupil:
            self.split_size_pupil = int(split_window_pupil * eyetracking_srate)
            self.split_window_pupil = split_window_pupil
            _, self.num_pupil_channels, self.num_windows_pupil, num_timepoints_per_window_pupil = sliding_window_view(x_pupil, window_shape=self.split_size_pupil,axis=2)[:, :, 0::self.split_size_pupil, :].shape
            cw_weights_pupil_folds = np.empty((num_folds, self.num_windows_pupil))
            roc_auc_folds_pupil = np.empty(num_folds)
            fpr_folds_pupil = []
            tpr_folds_pupil = []

        self.split_window_eeg = split_window_eeg
        self.exg_srate = exg_srate
        self.split_size_eeg = int(split_window_eeg * exg_srate)  # split data into 100ms bins

        # multi-fold cross-validation
        cross_val_folds = StratifiedShuffleSplit(n_splits=num_folds, random_state=random_seed)
        _, self.num_eeg_channels, self.num_windows_eeg, num_timepoints_per_window_eeg = sliding_window_view(x_eeg, window_shape=self.split_size_eeg, axis=2)[:, :, 0::self.split_size_eeg, :].shape
        cw_weights_eeg_folds = np.empty((num_folds, self.num_windows_eeg))
        activations_folds = np.empty((num_folds, len(self.event_names), self.num_eeg_channels, self.num_windows_eeg, num_timepoints_per_window_eeg))
        roc_auc_folds_eeg = np.empty(num_folds)
        fpr_folds_eeg = []
        tpr_folds_eeg = []


        # x_eeg_transformed, pca, ica = compute_pca_ica(x[0], num_top_compoenents)
        x_eeg_transformed = x_eeg_pca_ica
        best_auc = 0
        for i, (train, test) in enumerate(cross_val_folds.split(x_eeg, y)):  # cross-validation; group arguement is not necessary unless using grouped folds
            if verbose: print(f"Working on {i + 1} fold of {num_folds}")

            x_eeg_transformed_train, x_eeg_transformed_test, y_train, y_test = x_eeg_transformed[train],  x_eeg_transformed[test], y[train], y[test]
            x_eeg_transformed_train, y_train_eeg = rebalance_classes(x_eeg_transformed_train,y_train)  # rebalance by class
            if self._use_pupil:
                x_pupil_train, x_pupil_test, _, _ = x_pupil[train], x_pupil[test], y[train], y[test]
                x_pupil_train, y_train_pupil = rebalance_classes(x_pupil_train, y_train)  # rebalance by class
                assert np.all(y_train_eeg == y_train_pupil)

            y_train = y_train_eeg
            x_eeg_test = x_eeg[test]

            # x_eeg_transformed_train_windowed = sliding_window_view(x_eeg_transformed_train, window_shape=self.split_size_eeg,axis=2)[:, :, 0::self.split_size_eeg,:]  # shape = #trials, #channels, #windows, #time points per window
            # x_eeg_transformed_test_windowed = sliding_window_view(x_eeg_transformed_test, window_shape=self.split_size_eeg,axis=2)[:, :, 0::self.split_size_eeg,:]  # shape = #trials, #channels, #windows, #time points per window
            # x_eeg_test_windowed = sliding_window_view(x_eeg_test, window_shape=self.split_size_eeg, axis=2)[:, :, 0::self.split_size_eeg, :]  # shape = #trials, #channels, #windows, #time points per window

            x_eeg_transformed_train_windowed = self._split_by_window(x_eeg_transformed_train, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window
            x_eeg_transformed_test_windowed = self._split_by_window(x_eeg_transformed_test, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window
            x_eeg_test_windowed = self._split_by_window(x_eeg_test, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window

            num_train_trials, self.num_channels_eeg, self.num_windows_eeg, self.num_timepoints_per_window_eeg = x_eeg_transformed_train_windowed.shape

            # compute Fisher's LD for each temporal window
            if verbose >= 2: print("Computing windowed LDA per channel, and project per window and trial")
            lda_weights_eeg, projection_train_eeg, projection_test_eeg, window_eeg_ldas = _train_compute_window_projections(x_eeg_transformed_train_windowed, x_eeg_transformed_test_windowed, y_train)
            if verbose >= 2: print('Computing forward model from window projections for test set')
            activation = compute_forward(x_eeg_test_windowed, y_test, projection_test_eeg)
            # train classifier, use gradient descent to find the cross-window weights

            # x_pupil_train_windowed = sliding_window_view(x_pupil_train, window_shape=self.split_size_pupil, axis=2)[:, :,0::self.split_size_pupil,:]  # shape = #trials, #channels, #windows, #time points per window
            # x_pupil_test_windowed = sliding_window_view(x_pupil_test, window_shape=self.split_size_pupil, axis=2)[:, :,0::self.split_size_pupil,:]  # shape = #trials, #channels, #windows, #time points per window
            if self._use_pupil:
                x_pupil_train_windowed = self._split_by_window(x_pupil_train, self.split_size_pupil)  # shape = #trials, #channels, #windows, #time points per window
                x_pupil_test_windowed = self._split_by_window(x_pupil_test, self.split_size_pupil)  # shape = #trials, #channels, #windows, #time points per window
                _, self.num_channels_pupil, self.num_windows_pupil, self.num_timepoints_per_window_pupil = x_pupil_train_windowed.shape
                lda_weights_pupil, projection_train_pupil, projection_test_pupil, window_pupil_ldas = _train_compute_window_projections(x_pupil_train_windowed, x_pupil_test_windowed, y_train)
                # z-norm the projections
                projection_train_pupil, projection_test_pupil, self.pupil_mean, self.pupil_std = z_norm_projection(projection_train_pupil, projection_test_pupil)

            projection_train_eeg, projection_test_eeg, self.eeg_mean, self.eeg_std = z_norm_projection(projection_train_eeg, projection_test_eeg)

            if verbose >= 2: print('Solving cross bin weights')
            cw_weights_eeg, roc_auc_eeg, fpr_eeg, tpr_eeg, crossbin_model_eeg = solve_crossbin_weights(projection_train_eeg, projection_test_eeg,y_train, y_test, self.num_windows_eeg)

            if self._use_pupil:
                cw_weights_pupil, roc_auc_pupil, fpr_pupil, tpr_pupil, crossbin_model_pupil = solve_crossbin_weights( projection_train_pupil, projection_test_pupil, y_train, y_test, self.num_windows_pupil)
                projection_combined_train = np.concatenate([projection_train_eeg, projection_train_pupil], axis=1)
                projection_combined_test = np.concatenate([projection_test_eeg, projection_test_pupil], axis=1)
                cw_weights_combined, roc_auc_combined, fpr_combined, tpr_combined, crossbin_model_combined = solve_crossbin_weights(projection_combined_train, projection_combined_test, y_train, y_test,self.num_windows_pupil)
            else:
                roc_auc_pupil = None
                roc_auc_combined = roc_auc_eeg

            if roc_auc_combined > best_auc:  # save the weights of the best model across folds
                best_auc = roc_auc_combined
                self.window_eeg_ldas = window_eeg_ldas
                self.crossbin_model_eeg = crossbin_model_eeg
                if self._use_pupil:
                    self.window_pupil_ldas = window_pupil_ldas
                    self.crossbin_model_pupil = crossbin_model_pupil
                    self.crossbin_model_combined = crossbin_model_combined

            if self._use_pupil:
                cw_weights_pupil_folds[i] = cw_weights_pupil
                roc_auc_folds_pupil[i] = roc_auc_pupil
                fpr_folds_pupil.append(fpr_pupil)
                tpr_folds_pupil.append(tpr_pupil)

            cw_weights_eeg_folds[i] = cw_weights_eeg
            activations_folds[i] = activation

            roc_auc_folds_eeg[i] = roc_auc_eeg
            fpr_folds_eeg.append(fpr_eeg)
            tpr_folds_eeg.append(tpr_eeg)
            # print(f'Fold {i}, auc is {roc_auc_folds[i]}')

        if eeg_montage is not None:
            plot_forward(np.mean(activations_folds, axis=0), self.event_names, split_window_eeg, self.num_windows_eeg, exg_srate=exg_srate, notes=f"{notes} Average over {num_folds}-fold's test set", *args, **kwargs)

        if verbose:
            print(f"Mean EEG cross ROC-AUC is {np.mean(roc_auc_folds_eeg)}")
            if self._use_pupil:
                print(f"Mean Pupil cross ROC-AUC is {np.mean(roc_auc_folds_pupil)}")

        if is_plots:
            best_fold_i = np.argmax(roc_auc_folds_eeg)
            display = metrics.RocCurveDisplay(fpr=fpr_folds_eeg[best_fold_i], tpr=tpr_folds_eeg[best_fold_i],
                                              roc_auc=roc_auc_folds_eeg[best_fold_i],
                                              estimator_name='example estimator')
            fig = plt.figure(figsize=(10, 10))
            display.plot(ax=plt.gca(), name='ROC')
            plt.tight_layout()
            plt.title(f"{notes} EEG ROC of the best cross-val fold")
            plt.show()

            if self._use_pupil:
                best_fold_i = np.argmax(roc_auc_folds_pupil)
                display = metrics.RocCurveDisplay(fpr=fpr_folds_pupil[best_fold_i], tpr=tpr_folds_pupil[best_fold_i],
                                                  roc_auc=roc_auc_folds_pupil[best_fold_i],
                                                  estimator_name='example estimator')
                fig = plt.figure(figsize=(10, 10))
                display.plot(ax=plt.gca(), name='ROC')
                plt.tight_layout()
                plt.title(f"{notes} Pupil ROC of the best cross-val fold")
                plt.show()

            fig = plt.figure(figsize=(15, 10))
            plt.boxplot(cw_weights_eeg_folds)
            # plt.plot(cross_window_weights)
            x_labels = [f"{int((i - 1) * split_window_eeg * 1e3)}ms" for i in range(self.num_windows_eeg)]
            x_ticks = np.arange(0.5, self.num_windows_eeg + 0.5, 1)
            plt.plot(list(range(1, self.num_windows_eeg + 1)), np.mean(cw_weights_eeg_folds, axis=0), label="folds average")

            plt.xticks(ticks=x_ticks, labels=x_labels)
            plt.xlabel("100 ms windowed bins")
            plt.ylabel("Cross-bin weights")
            plt.title(f'{notes} Cross-bin weights, {num_folds}-fold cross validation')
            plt.legend()
            plt.tight_layout()
            plt.show()

            if self._use_pupil:
                fig = plt.figure(figsize=(15, 10))
                plt.boxplot(cw_weights_pupil_folds)
                # plt.plot(cross_window_weights)
                x_labels = [f"{int((i - 1) * split_window_pupil * 1e3)}ms" for i in range(self.num_windows_pupil)]
                x_ticks = np.arange(0.5, self.num_windows_pupil + 0.5, 1)
                plt.plot(list(range(1, self.num_windows_pupil + 1)), np.mean(cw_weights_pupil_folds, axis=0),
                         label="folds average")

                plt.xticks(ticks=x_ticks, labels=x_labels)
                plt.xlabel("500 ms windowed bins")
                plt.ylabel("Cross-bin weights")
                plt.title(f'{notes} Cross-bin weights, {num_folds}-fold cross validation')
                plt.legend()
                plt.tight_layout()
                plt.show()

        return roc_auc_combined, roc_auc_eeg, roc_auc_pupil

    def eval(self, x_eeg, x_eeg_pca_ica, x_pupil, y, notes="", eeg_montage=None, *args, **kwargs):
        y = self._encoder(np.copy(y))

        x_eeg_transformed_windowed = self._split_by_window(x_eeg_pca_ica, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window
        x_eeg_windowed = self._split_by_window(x_eeg, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window

        if self._use_pupil:
            x_pupil_windowed = self._split_by_window(x_pupil, self.split_size_pupil)  # shape = #trials, #channels, #windows, #time points per window
            lda_projections_pupil = _compute_window_lda_projections(x_pupil_windowed, self.window_pupil_ldas)
            projection_pupil = z_norm_hdca(lda_projections_pupil, self.pupil_mean, self.pupil_std)

        lda_projections_eeg = _compute_window_lda_projections(x_eeg_transformed_windowed, self.window_eeg_ldas)
        activation = compute_forward(x_eeg_windowed, y, lda_projections_eeg)

        projection_eeg = z_norm_hdca(lda_projections_eeg, self.eeg_mean, self.eeg_std)

        y_pred_eeg, roc_auc_eeg, fpr_eeg, tpr_eeg = eval_crossbin_model(projection_eeg, y, self.crossbin_model_eeg)

        if self._use_pupil:
            projection_combined = np.concatenate([projection_eeg, projection_pupil], axis=1)
            _, roc_auc_pupil, fpr_pupil, tpr_pupil = eval_crossbin_model(projection_pupil, y, self.crossbin_model_pupil)
            y_pred, roc_auc_combined, fpr_combined, tpr_combined = eval_crossbin_model(projection_combined, y, self.crossbin_model_combined)
        else:
            roc_auc_combined, roc_auc_pupil = None, None
            y_pred = y_pred_eeg
        if eeg_montage is not None:
            plot_forward(activation, self.event_names, self.split_window_eeg, self.num_windows_eeg,exg_srate=self.exg_srate, notes=f"{notes} Forward model activation", eeg_montage=eeg_montage, *args, **kwargs)
        return y_pred, roc_auc_combined, roc_auc_eeg, roc_auc_pupil


    def transform(self, x_eeg_pca_ica, x_pupil=None):
        x_eeg_transformed_windowed = self._split_by_window(x_eeg_pca_ica, self.split_size_eeg)  # shape = #trials, #channels, #windows, #time points per window

        lda_projections_eeg = _compute_window_lda_projections(x_eeg_transformed_windowed, self.window_eeg_ldas)
        projection_eeg = z_norm_hdca(lda_projections_eeg, self.eeg_mean, self.eeg_std)

        if self._use_pupil:
            x_pupil_windowed = self._split_by_window(x_pupil, self.split_size_pupil)  # shape = #trials, #channels, #windows, #time points per window
            lda_projections_pupil = _compute_window_lda_projections(x_pupil_windowed, self.window_pupil_ldas)
            projection_pupil = z_norm_hdca(lda_projections_pupil, self.pupil_mean, self.pupil_std)
            projection_combined = np.concatenate([projection_eeg, projection_pupil], axis=1)
            y_pred = self.crossbin_model_combined.predict(projection_combined)
        else:
            y_pred = self.crossbin_model_eeg.predict(projection_eeg)
        return y_pred


    def _split_by_window(self, data, window_shape):
        return  sliding_window_view(data, window_shape=window_shape, axis=2)[:, :,0::window_shape, :]  # shape = #trials, #channels, #windows, #time points per window

