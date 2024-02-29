import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript

from physiolabxr.utils.buffers import DataBuffer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from enum import Enum

Plot = True
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except:
    Plot = False
    print("Seaborn and Matplotlib not installed. Skip Plot.")

# configurations
eeg_stream_name = 'UnicornHybridBlackLSL'

# eeg channel names
eeg_channel_names = [
    "EEG1", "EEG2", "EEG3", "EEG4", "EEG5", "EEG6", "EEG7", "EEG8",
    "Accelerometer X", "Accelerometer Y", "Accelerometer Z",
    "Gyroscope X", "Gyroscope Y", "Gyroscope Z",
    "Battery Level",
    "Counter",
    "Validation Indicator",
    "Timestamp",
    "Marker"
]

# epoch configuration
eeg_epoch_t_min = -0.2
eeg_epoch_t_max = 1.0

# unicorn hybrid black sampling rate
eeg_sampling_rate = 250

# get the sample before and after the trigger event
sample_before_epoch = np.floor(eeg_epoch_t_min * eeg_sampling_rate).astype(int)
sample_after_epoch = np.floor(eeg_epoch_t_max * eeg_sampling_rate).astype(int)

# eeg channel number and index
eeg_channel_index = [0, 1, 2, 3, 4, 5, 6, 7]

# event marker stream name for the Unity P300 Speller
event_marker_stream_name = 'CortexTypeP300SpellerEventMarkerLSL'

# the board configuration for the P300 Speller
Board = [['A', 'B', 'C', 'D', 'E', 'F'],
         ['G', 'H', 'I', 'J', 'K', 'L'],
         ['M', 'N', 'O', 'P', 'Q', 'R'],
         ['S', 'T', 'U', 'V', 'W', 'X'],
         ['Y', 'Z', '0', '1', '2', '3'],
         ['4', '5', '6', '7', '8', '9']]


class IndexClass(int, Enum):
    pass


class EventMarkerChannelInfo(IndexClass):
    FlashingBlockMarker = 0
    FlashingMarker = 1
    FlashingRowOrColumnMarker = 2
    FlashingRowOrColumnIndexMarker = 3
    FlashingTargetMarker = 4


class CortexType(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression()

        self.train_x = []
        self.train_y = []
        self.run_function = False

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # print('Loop function is called')
        pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
        self.inputs.clear_buffer_data()

    # RPC Calls

    def get_train_trail_epochs(self):
        '''
        Extracts EEG epochs for each flashing target event from the input EEG and event marker streams, and appends them to the training dataset.
        This function iterates through each event identified in the event marker stream. If the event is a flashing target event,
        it locates the corresponding EEG data timestamp and extracts a specific epoch window around this event.
        The epoch window ranges from 0.2 seconds before to 1 second after the event timestamp,
        corresponding to a specific number of samples before and after the event in the EEG data. Each extracted epoch,
        along with its associated label (indicating whether the target was a flashing target),
        is appended to the training datasets (`train_x` for the EEG data, `train_y` for the labels).

        Note: we do not need to use the row and column markers, as we are only use the flashing target markers to train the binary classifier.
              Additinally, we assume the eeg data is already preprocessed and filtered.
              Refer to "https://physiolabxrdocs.readthedocs.io/en/latest/DSP.html" for more information on EEG preprocessing and filtering.
        '''

        # get the trail events
        events = self.inputs.get_data(event_marker_stream_name)
        events_timestamps = self.inputs.get_timestamps(event_marker_stream_name)

        eeg_timestamps = self.inputs.get_timestamps(eeg_stream_name)
        eeg_data = self.inputs.get_data(eeg_stream_name)[eeg_channel_index, :]

        number_of_epochs = 0

        # get the eeg epoch data for each flashing target event
        for event, events_timestamp in zip(events.T,
                                           events_timestamps):  # the transpose makes it easier to iterate over the events
            # if the event is a flashing target event
            if event[EventMarkerChannelInfo.FlashingMarker] == 1:
                # find the eeg time stamp index with the event time stamp using np.searchsorted
                eeg_timestamp_index = np.searchsorted(eeg_timestamps, events_timestamp)
                # get the eeg epoch data with the event timestamp, -0.2s to 1s which is 50 samples before and 250 samples after the event timestamp
                eeg_epoch = eeg_data[:,
                            eeg_timestamp_index + sample_before_epoch:eeg_timestamp_index + sample_after_epoch]
                label = event[EventMarkerChannelInfo.FlashingTargetMarker]

                self.train_x.append(eeg_epoch)
                self.train_y.append(label)

                number_of_epochs += 1

        # print how many epochs were found
        print(f"Found {number_of_epochs} epochs")

        # clear the buffer data. This is important to avoid counting the same epochs again
        self.inputs.clear_buffer_data()

    def train_epochs(self):

        # convert the train data to numpy array
        X = np.array(self.train_x)
        y = np.array(self.train_y)

        # split the data into train and test set
        test_size = 0.2

        # split the data into train and test set
        x_train, x_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size)

        # rebalance classes
        rebalance_classes(x_train, y_train, by_channel=True)

        # Reshape the data. This is simple logistic regression, so we flatten the input x over the channel axis
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # train the model
        self.model.fit(x_train, y_train)

        # predict the test set
        y_pred = self.model.predict(x_test)

        # print the confusion matrix
        confusion_matrix(y_test, y_pred)



    def predict(self):

        probability_matrix = np.zeros(shape=np.array(Board).shape)

        # get the trail events
        events = self.inputs.get_data(event_marker_stream_name)
        events_timestamps = self.inputs.get_timestamps(event_marker_stream_name)

        eeg_timestamps = self.inputs.get_timestamps(eeg_stream_name)
        eeg_data = self.inputs.get_data(eeg_stream_name)[eeg_channel_index, :]

        # get the eeg epoch data for each flashing target event
        for event, events_timestamp in zip(events.T, events_timestamps):

            # if the event is a flashing target event
            if event[EventMarkerChannelInfo.FlashingMarker] == 1:
                # find the eeg time stamp index with the event time stamp using np.searchsorted
                eeg_timestamp_index = np.searchsorted(eeg_timestamps, events_timestamp)
                # get the eeg epoch data with the event timestamp, -0.2s to 1s which is 50 samples before and 250 samples after the event timestamp
                eeg_epoch = eeg_data[:,
                            eeg_timestamp_index + sample_before_epoch:eeg_timestamp_index + sample_after_epoch]

                # flatten the eeg epoch to fit the model
                x = eeg_epoch.flatten()
                y = self.model.predict_proba([x])[0][1]

                # the FlashingRowOrColumnIndexMarker will tell us the row or column index of the flashing item
                row_or_column_index = event[EventMarkerChannelInfo.FlashingRowOrColumnIndexMarker].astype(int)

                # the FlashingRowOrColumnMarker will tell us if the flashing item is a row or column, then we accumulate the probability on the row or column
                if event[EventMarkerChannelInfo.FlashingRowOrColumnMarker] == 1:
                    probability_matrix[row_or_column_index, :] += y
                else:
                    probability_matrix[:, row_or_column_index] += y

        # print the probability matrix
        print(f"Probability Matrix: {probability_matrix}")
        # the grid with the maximum probability will be the predicted letter
        max_probability_letter_index = np.unravel_index(probability_matrix.argmax(), probability_matrix.shape)
        print(f"Max Probability Letter Index: {max_probability_letter_index}")
        # get the letter from the board, max_probability_letter_index is a tuple of the row and column index
        predicted_letter = Board[max_probability_letter_index[0]][max_probability_letter_index[1]]
        print(f"Predicted Letter: {predicted_letter}")

        self.inputs.clear_buffer_data()
        return predicted_letter

    def clear_train_data_buffer(self):

        self.train_x = []
        self.train_y = []


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
    print("Confusion Matrix:")
    print(cm)
    score = f1_score(y_test, y_pred, average='macro')
    print("F1 Score (Macro): {:.2f}".format(score))

    if Plot:
        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        all_sample_title = 'Accuracy Score: {0}'.format(score)
        plt.title(all_sample_title, size=15)
        plt.show()


