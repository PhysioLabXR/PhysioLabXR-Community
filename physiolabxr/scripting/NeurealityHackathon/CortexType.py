import numpy as np
from pylsl import StreamInfo, StreamOutlet

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
    FlashingTrailMarker = 0
    FlashingMarker = 1
    FlashingRowOrColumnMarker = 2
    FlashingRowOrColumnIndexMarker = 3
    FlashingTargetMarker = 4

    TrainMarkerIndex = 5
    TestMarkerIndex = 6

    InterruptMarkerIndex = 7


class CurrentState(IndexClass):
    IdleState = 0
    TrainState = 1
    TestState = 2


class CortexType(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

        self.current_state = CurrentState.IdleState  # the initial state is idle

        self.model = LogisticRegression()

        self.data_buffer = DataBuffer()

        self.train_x = []
        self.train_y = []

        ########################
        # create a lsl stream outlet that to send the index of the predicted letter index to the Unity P300 Speller
        info = StreamInfo('CortexTypePredictionLSL',
                          'PredictionIndex',
                          2, # the number of channels is 2 because we are sending the row and column index of the predicted letter
                          1,
                          'float32',
                          'CortexTypePredictedLetterIndex')
        self.prediction_letter_index_outlet = StreamOutlet(info)
        ########################

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # print('Loop function is called')

        if event_marker_stream_name not in self.inputs.keys() or eeg_stream_name not in self.inputs.keys():
            print('No EEG stream or no event marker stream, return')
            self.current_state = CurrentState.IdleState
            return

        event_marker_data = self.inputs.get_data(event_marker_stream_name)
        event_marker_timestamps = self.inputs.get_timestamps(event_marker_stream_name)

        for event_index, event_marker_timestamp in enumerate(event_marker_timestamps):
            event_marker = event_marker_data[:, event_index]
            FlashingTrailMarker = event_marker[EventMarkerChannelInfo.FlashingTrailMarker]
            # FlashingMarker = event_marker[EventMarkerChannelInfo.FlashingMarker]
            # FlashingRowOrColumnMarker = event_marker[EventMarkerChannelInfo.FlashingRowOrColumnMarker]
            # FlashingRowOrColumnIndexMarker = event_marker[EventMarkerChannelInfo.FlashingRowOrColumnIndexMarker]
            # FlashingTargetMarker = event_marker[EventMarkerChannelInfo.FlashingTargetMarker]

            TrainMarker = event_marker[EventMarkerChannelInfo.TrainMarkerIndex]
            TestMarker = event_marker[EventMarkerChannelInfo.TestMarkerIndex]

            InterruptMarker = event_marker[EventMarkerChannelInfo.InterruptMarkerIndex]

            if FlashingTrailMarker == 1:
                self.current_state = CurrentState.TrainState
            elif FlashingTrailMarker == -1:
                self.get_train_trail_epochs()
                self.current_state = CurrentState.IdleState
            elif FlashingTrailMarker == 2:
                self.current_state = CurrentState.TestState
            elif FlashingTrailMarker == -2:
                self.current_state = CurrentState.IdleState

            if TrainMarker == 1:
                self.train_epochs()
                self.clear_train_data_buffer()

            if InterruptMarker == 1:
                self.current_state = CurrentState.IdleState
                self.inputs.clear_buffer_data()
                self.data_buffer.clear_buffer_data()
                self.clear_train_data_buffer()


            if TestMarker == 1:
                max_probability_letter_index = self.predict()

                self.prediction_letter_index_outlet.push_sample(max_probability_letter_index)

        if self.current_state == CurrentState.TrainState or self.current_state == CurrentState.TestState:
            self.data_buffer.update_buffers(self.inputs.buffer)
            print('Data buffer updated')

        self.inputs.clear_buffer_data()

        pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
        self.inputs.clear_buffer_data()

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

        print('Get train trail epochs')
        # get the trail events
        events = self.data_buffer.get_data(event_marker_stream_name)
        events_timestamps = self.data_buffer.get_timestamps(event_marker_stream_name)

        eeg_timestamps = self.data_buffer.get_timestamps(eeg_stream_name)
        eeg_data = self.data_buffer.get_data(eeg_stream_name)[eeg_channel_index, :]

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
        self.data_buffer.clear_buffer_data()

    def train_epochs(self):
        """
           Trains a logistic regression model on EEG epoch data for a classification task. The process involves preparing
           the training data, splitting it into training and testing sets, rebalancing the classes to address any imbalance,
           reshaping the data to fit the logistic regression model, training the model on the training set, and evaluating
           its performance on the testing set using a confusion matrix.

           Steps involved in the training process:
           1. Converts the training inputs and labels into numpy arrays for compatibility with machine learning libraries.
           2. Splits the data into training and testing sets, with a portion of the data specified by `test_size` reserved
              for testing.
           3. Optionally rebalances the classes in the training set to ensure equal representation of all classes,
              particularly useful in cases where some classes are underrepresented.
           4. Reshapes the training and testing input data by flattening the input over the channel axis, making it suitable
              for logistic regression which expects 2D input.
           5. Trains the logistic regression model on the reshaped training data.
           6. Predicts the labels for the testing set using the trained model.
           7. Prints the confusion matrix to evaluate the model's performance, showing how the model's predictions compare to
              the actual labels.

           Note:
               - The function assumes `self.train_x` and `self.train_y` are populated with the EEG epoch data and their
                 corresponding labels, respectively.
               - It uses `train_test_split` from sklearn to split the data, assumes the presence of a `rebalance_classes`
                 function to handle class imbalance, and requires a `self.model` attribute that supports the `fit` and
                 `predict` methods.
               - The performance evaluation is done through a confusion matrix, which helps in understanding the model's
                 classification accuracy across different classes.

           This function does not return any value but modifies the model in place and prints the confusion matrix for
           immediate evaluation of the model's performance.
           """

        print('Train epochs')
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
        """
        Predicts the most likely target letter from a series of EEG (electroencephalogram) signal epochs associated with
        flashing events in a P300 speller matrix. The function processes EEG data and event markers to compute a
        probability matrix indicating the likelihood of each cell in the speller matrix being the target. The method involves
        identifying EEG epochs corresponding to flashing events, extracting features, applying a pre-trained model to
        predict the probability of each event corresponding to the target, and accumulating these probabilities across
        rows and columns based on the nature of the flashing event (row or column flash). The cell in the speller matrix
        with the highest accumulated probability is determined to be the target, and its associated letter is predicted.

        Steps involved in the prediction process:
        1. Initializes a probability matrix corresponding to the layout of the P300 speller matrix.
        2. Retrieves the event markers and their timestamps to identify flashing target events.
        3. For each flashing event, finds the corresponding EEG epoch using the event timestamp.
        4. Flattens the EEG epoch data and uses a pre-trained model to predict the probability of the target for the epoch.
        5. Accumulates these probabilities in the probability matrix based on whether the flash was for a row or column.
        6. Identifies the cell with the maximum probability as the predicted target and retrieves its associated letter.

        The function clears the buffer data after each prediction to prepare for the next set of inputs.

        Returns:
            predicted_letter (str): The letter from the P300 speller matrix predicted to be the target based on the
                                    accumulated probabilities.

        Note:
            - This function assumes the existence of a pre-trained model and appropriate methods for data retrieval and
              buffer clearing (`self.data_buffer.get_data`, `self.data_buffer.get_timestamps`, `self.data_buffer.clear_buffer_data`).
            - The function relies on several global variables and constants, such as `Board`, `EventMarkerChannelInfo`,
              `eeg_stream_name`, `event_marker_stream_name`, `eeg_channel_index`, `sample_before_epoch`, and
              `sample_after_epoch`, which should be defined and properly configured in the surrounding context.
        """

        print('Predict')
        probability_matrix = np.zeros(shape=np.array(Board).shape)

        # get the trail events
        events = self.data_buffer.get_data(event_marker_stream_name)
        events_timestamps = self.data_buffer.get_timestamps(event_marker_stream_name)

        eeg_timestamps = self.data_buffer.get_timestamps(eeg_stream_name)
        eeg_data = self.data_buffer.get_data(eeg_stream_name)[eeg_channel_index, :]

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

        self.data_buffer.clear_buffer_data()

        return max_probability_letter_index

    def clear_train_data_buffer(self):
        # clear the buffer data. This is important to avoid counting the same epochs again
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
