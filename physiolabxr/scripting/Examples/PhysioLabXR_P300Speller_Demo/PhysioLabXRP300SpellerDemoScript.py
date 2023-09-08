import numpy as np
from physiolabxr.scripting.Examples.PhysioLabXR_P300Speller_Demo.PhysioLabXRP300SpellerDemoConfig import *
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
import seaborn as sns

class PhysioLabXRGameP300SpellerDemoScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # test network
        self.EXPERIMENT_STATE = ExperimentStateMarker.StartState
        self.IN_FLASHING_BLOCK = False
        self.model = LogisticRegression()
        self.data_buffer = DataBuffer()

        self.train_state_x = []
        self.train_state_y = []

        # self.test_state_x = []
        # self.test_state_y = []

        self.StateEnterExitMarker = 0
        self.FlashBlockStartEndMarker = 0
        self.FlashingMarker = 0
        self.FlashingItemIndexMarker = 0
        self.FlashingTargetMarker = 0
        self.StateInterruptMarker = 0



    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # print('Loop function is called')
        if EEG_STREAM_NAME not in self.inputs.keys() or EVENT_MARKER_CHANNEL_NAME not in self.inputs.keys():
            # if no event marker or no eeg stream, we do not do anything
            print('No EEG stream or no event marker stream, return')
            # state is None, and Flashing is False. We interrupt the experiment
            self.EXPERIMENT_STATE = None
            self.IN_FLASHING_BLOCK = False
            return

        event_marker_data = self.inputs.get_data(EVENT_MARKER_CHANNEL_NAME)
        event_marker_timestamps = self.inputs.get_timestamps(EVENT_MARKER_CHANNEL_NAME)
        # print(event_marker_data)
        # in this example, we only care about the Train, Test, Interrupt, and Block Start/Block end markers
        # process event markers
        # try:
        for event_index, event_marker_timestamp in enumerate(event_marker_timestamps):
            event_marker = event_marker_data[:, event_index]

            self.StateEnterExitMarker = event_marker[EventMarkerChannelInfo.StateEnterExitMarker]
            self.FlashBlockStartEndMarker = event_marker[EventMarkerChannelInfo.FlashBlockStartEndMarker]
            self.FlashingMarker = event_marker[EventMarkerChannelInfo.FlashingMarker]
            self.FlashingItemIndexMarker = event_marker[EventMarkerChannelInfo.FlashingItemIndexMarker]
            self.FlashingTargetMarker = event_marker[EventMarkerChannelInfo.FlashingTargetMarker]
            self.StateInterruptMarker = event_marker[EventMarkerChannelInfo.StateInterruptMarker]

            if self.StateInterruptMarker:
                # state is None, and Flashing is False. We interrupt the experiment
                self.EXPERIMENT_STATE = None
                self.IN_FLASHING_BLOCK = False

            elif self.StateEnterExitMarker:
                self.switch_state(self.StateEnterExitMarker)

            elif self.FlashBlockStartEndMarker:
                print('Block Start/End Marker: ', self.FlashBlockStartEndMarker)

                if self.FlashBlockStartEndMarker == 1:  # flash start
                    self.IN_FLASHING_BLOCK = True
                    print('Start Flashing Block')
                    self.inputs.clear_up_to(event_marker_timestamp)
                    # self.data_buffer.update_buffers(self.inputs.buffer)
                if self.FlashBlockStartEndMarker == -1:  # flash end
                    self.IN_FLASHING_BLOCK = False
                    print('End Flashing Block')
                    if self.EXPERIMENT_STATE == ExperimentStateMarker.TrainState:
                        # train callback
                        self.train_callback()
                        pass
                    elif self.EXPERIMENT_STATE == ExperimentStateMarker.TestState:
                        # test callback
                        self.test_callback()
                        pass
            elif self.FlashingMarker:  # flashing
                print('Flashing Marker: ', self.FlashingMarker)
                print('Flashing Target Marker: ', self.FlashingTargetMarker)
                print('Flashing Item Index Marker: ', self.FlashingItemIndexMarker)
            else:
                pass
        # except Exception as e:
        #     print(e)
        #     return

        # if flashing
        if self.IN_FLASHING_BLOCK:
            # the event marker in the buffer only contains the event marker in the current flashing block
            self.data_buffer.update_buffers(self.inputs.buffer)
            # print('In Flashing Block, save data to buffer')

        self.inputs.clear_buffer_data()


    def switch_state(self, new_state):
        if new_state == ExperimentStateMarker.StartState:
            print('Start State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.StartState

        elif new_state == ExperimentStateMarker.TrainIntroductionState:
            print('Train Introduction State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TrainIntroductionState

        elif new_state == ExperimentStateMarker.TrainState:
            print('Train State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TrainState

        elif new_state == ExperimentStateMarker.TestIntroductionState:
            print('Test Introduction State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TestIntroductionState

        elif new_state == ExperimentStateMarker.TestState:
            print('Test State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.TestState

        elif new_state == ExperimentStateMarker.EndState:
            print('End State')
            self.EXPERIMENT_STATE = ExperimentStateMarker.EndState

        else:
            print('Exit Current State: ', new_state)
            self.EXPERIMENT_STATE = None

    def train_callback(self):
        # train callback


        flash_timestamps = self.data_buffer.get_timestamps(EVENT_MARKER_CHANNEL_NAME)
        eeg_timestamps = self.data_buffer.get_timestamps(EEG_STREAM_NAME)
        eeg_epoch_start_indices = np.searchsorted(eeg_timestamps, flash_timestamps, side='left')

        sample_before_epoch = np.floor(EEG_EPOCH_T_MIN * EEG_SAMPLING_RATE).astype(int)
        sample_after_epoch = np.floor(EEG_EPOCH_T_MAX * EEG_SAMPLING_RATE).astype(int)
        for eeg_epoch_start_index in eeg_epoch_start_indices:
            eeg_epoch = self.data_buffer.get_data(EEG_STREAM_NAME)[:, eeg_epoch_start_index+sample_before_epoch:eeg_epoch_start_index+sample_after_epoch]
            self.train_state_x.append(eeg_epoch)

        labels = self.data_buffer.get_data(EVENT_MARKER_CHANNEL_NAME)[EventMarkerChannelInfo.FlashingTargetMarker, :]
        self.train_state_y.extend(labels)

        # train based on all the data in the buffer
        x = np.array(self.train_state_x)
        y = np.array(self.train_state_y)
        print("Train On Data: ", x.shape, y.shape)
        train_logistic_regression(x, y, self.model, test_size=0.1)
        self.data_buffer.clear_buffer_data() # clear the buffer data for the next flashing block
        pass

    def test_callback(self):
        # test callback

        x = []

        flash_timestamps = self.data_buffer.get_timestamps(EVENT_MARKER_CHANNEL_NAME)
        eeg_timestamps = self.data_buffer.get_timestamps(EEG_STREAM_NAME)
        eeg_epoch_start_indices = np.searchsorted(eeg_timestamps, flash_timestamps, side='left')

        sample_before_epoch = np.floor(EEG_EPOCH_T_MIN * EEG_SAMPLING_RATE).astype(int)
        sample_after_epoch = np.floor(EEG_EPOCH_T_MAX * EEG_SAMPLING_RATE).astype(int)

        for eeg_epoch_start_index in eeg_epoch_start_indices:
            eeg_epoch = self.data_buffer.get_data(EEG_STREAM_NAME)[:, eeg_epoch_start_index+sample_before_epoch:eeg_epoch_start_index+sample_after_epoch]
            x.append(eeg_epoch)

        # predict based on all the data in the buffer
        x = np.array(x)
        x = x.reshape(x.shape[0], -1)
        y_target_probabilities = self.model.predict_proba(x)[:, 1]
        print(y_target_probabilities)
        flashing_item_indices = self.data_buffer.get_data(EVENT_MARKER_CHANNEL_NAME)[EventMarkerChannelInfo.FlashingItemIndexMarker, :]
        flashing_item_indices = np.array(flashing_item_indices).astype(int)
        probability_matrix = np.zeros(shape=np.array(Board).shape)
        for flashing_item_index, y_target_probability in zip(flashing_item_indices, y_target_probabilities):
            if flashing_item_index<=5: # this is row index
                row_index = flashing_item_index
                probability_matrix[row_index, :] += y_target_probability
            else: # this is column index, we need -6 to get the column index
                column_index = flashing_item_index-6
                probability_matrix[:, column_index] += y_target_probability

        # normalize the probability matrix to 0 to 1
        probability_matrix = probability_matrix / len(flashing_item_indices/24)



        print(probability_matrix)
        plt.imshow(probability_matrix, cmap='hot', interpolation='nearest')
        plt.show()

        self.set_output(PREDICTION_PROBABILITY_CHANNEL_NAME, probability_matrix.flatten())
        print("Prediction Probability Sent")

        self.data_buffer.clear_buffer_data()


    # cleanup is called when the stop button is hit
    def cleanup(self):
        self.model = None
        print('Cleanup function is called')

def train_logistic_regression(X, y, model, test_size=0.2):
    """
    Trains a logistic regression model on the input data and prints the confusion matrix.

    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target variable.
        model (LogisticRegression): An instance of LogisticRegression from scikit-learn.
        test_size (float): Proportion of the data to reserve for testing. Default is 0.2.

    Returns:
        None.

    Raises:
        TypeError: If model is not an instance of LogisticRegression.
        ValueError: If test_size is not between 0 and 1.

    """
    # Check if model is an instance of LogisticRegression
    if not isinstance(model, LogisticRegression):
        raise TypeError("model must be an instance of LogisticRegression.")

    # Check if test_size is between 0 and 1
    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)
    rebalance_classes(x_train, y_train, by_channel=True)

    # Reshape the data. This is simple logistic regression, so we flatten the input x
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # Fit the model to the training data and make predictions on the test data
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Print the confusion matrix
    confusion_matrix(y_test, y_pred)

def confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix for the predicted vs. actual labels and prints the accuracy score.

    Args:
        y_test (np.ndarray): Actual labels of the test set.
        y_pred (np.ndarray): Predicted labels of the test set.

    Returns:
        None.

    Raises:
        TypeError: If either y_test or y_pred are not numpy arrays.

    """
    # Check if y_test and y_pred are numpy arrays
    if not isinstance(y_test, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_test and y_pred must be numpy arrays.")

    # Calculate the confusion matrix and f1 score
    cm = metrics.confusion_matrix(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size=15)
    plt.show()

def rebalance_classes(x, y, by_channel=False):
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
            sm = SMOTE(k_neighbors=5, random_state=42)
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
        sm = SMOTE(random_state=42)
        x, y = sm.fit_resample(x, y)

        # Reshape the input data back to its original shape
        x = np.reshape(x, newshape=(len(x),) + epoch_shape)

    return x, y

# END CLASS