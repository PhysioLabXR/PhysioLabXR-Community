from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.rpc.decorator import rpc, async_rpc

class NeuralCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]         #defining frequency bands for filter bank
        self.mSequence = []                                     #creating a list of m_sequences
        self.frequency = 300                                    #default frequency of DSI-24
        self.data = DataBuffer()                                #generating a data buffer for EEG data
        self.cca_models = []                                    #creating a list to store all of the CCA models
        self.decoded_choices = []                               #creating a list to store all of the decoded choices
        self.mSequence = [
            [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0],  #mSequence1
            [1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],  #mSequence2
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1]   #mSequence3
        ]
        self.seq1_data = np.array([])
        self.seq2_data = np.array([])
        self.seq3_data = np.array([])
    def loop(self):
        EEG_Data = {                                            #creating a dictionary for EEG data
            'stream_name': 'EEG Data',                          #defining the stream name
            'frames': self.inputs['DSI-24'][0][14:18, :],       #choosing the correct channels
            'timestamps': self.inputs['DSI-24'][1]              #defining the timestamps
        }
        Sequence_Data = {}                                      #creating a dictionary for sequence data
        self.data.update_buffer(EEG_Data)                       #updating the data buffer with EEG data
        if self.data.get_data('EEG Data').shape[1] > 60000:     #if the data is longer than 200 seconds then cut off beginning of data so that it is to 200 seconds
            self.data.clear_stream_up_to_index(self, stream_name= 'EEG Data', cut_to_index= len(self.data.get_data('EEG Data').shape[1])-60000)
        if len(self.cca_models) == 3:                           #if training is complete (i.e there are 3 CCA models) then we can start decoding everything asyncronously
            self.decode_choice()                                  #adding the detected choice to the list of detected choices
    def cleanup(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]
        self.mSequence = []
        self.frequency = 300
        self.data = DataBuffer()
        self.cca_models = []
        self.decoded_choices = []
        self.mSequence = []
        return


    ##Basic Tools
#===================================================================================================
    def apply_filter_banks(self, data):
        band_data = {}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            band_data[band_key] = self.bandpass_filter(data, band[0], band[1], self.frequency)
        return band_data

    def bandpass_filter(self, data, lowcut, highcut,fs, order=8):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_data= filtfilt(b, a, data)
        return filtered_data



    ##Begin Training
#===================================================================================================
    #Need to add an RPC function that adds to training dataset
    @async_rpc
    def training(self) -> int:
        """
        Args:
            input0: int - 1 for choice 1, 2 for choice 2, 3 for choice 3
        Returns: Generates correlation coefficients for EEG data x m-sequence
        """
        # Train the CCA
        self.train_cca()                                                #start training the CCA
        return 1


    @async_rpc
    def add_seq_data(self, sequence_num: int): #Data is going to come in sequencially seq1 -> seq2 -> seq3 repeat
        if sequence_num == 0:
            self.seq1_data = np.append(self.seq1_data, self.data.get_data('EEG Data')[:,-1500:]) #Every 5 seconds
        elif sequence_num == 1:
            self.seq2_data = np.append(self.seq2_data, self.data.get_data('EEG Data')[:,-1500:])
        elif sequence_num == 2:
            self.seq3_data = np.append(self.seq3_data, self.data.get_data('EEG Data')[:,-1500:])

    def train_cca(self):
        """
        Trains the CCA model.
        This method generates spatial filters and templates for each target m-sequence.
        """

        segment_Length = 1500
        # Split data into segments for each m-sequence
        seq1_segments = np.array_split(self.seq1_data, self.seq1_data.shape[1] // segment_Length, axis=1)
        seq2_segments = np.array_split(self.seq2_data, self.seq2_data.shape[1] // segment_Length, axis=1)
        seq3_segments = np.array_split(self.seq3_data, self.seq3_data.shape[1] // segment_Length, axis=1)

        # Generate templates by averaging segments for each m-sequence
        templates = {
            1: np.mean(seq1_segments, axis=0),
            2: np.mean(seq2_segments, axis=0),
            3: np.mean(seq3_segments, axis=0)
        }

        # Filter the data using predefined frequency bands
        band_data = self.apply_filter_banks(self.data.get_data('EEG Data'))

        # Generate CCA-based spatial filters and templates for each band and each m-sequence
        cca_model = {}
        for band in self.freq_bands:
            band_key = f'band_{band[0]}_{band[1]}'
            cca_model[band_key] = {}

            for i in range(1, 4):  # Assuming there are 3 m-sequences
                cca = CCA(n_components=1)
                filtered_template = self.bandpass_filter(templates[i], band[0], band[1], self.frequency)
                cca.fit(filtered_template.T, self.mSequence[i - 1])
                cca_model[band_key][i] = cca

        # Store the CCA models
        self.cca_models = cca_model
        self.templates = templates  # Store templates for future use


    ##Begin Playing
#===================================================================================================
    @async_rpc
    def decode(self) -> int:
        # Get the choices decoded so far
        choices = self.decoded_choices

        # Determine the most common choice
        user_choice = max(set(choices), key=choices.count)

        # Clear the decoded choices list for the next round
        self.decoded_choices = []

        # Return the most common detected choice
        return user_choice
    def decode_choice(self):
        band_data = self.apply_filter_banks(self.data.get_data('EEG Data')[:, -1500])  # applying filter banks to EEG data
        self.correlation_coefficients = self.apply_shifting_window_cca(band_data)  # getting the correlation coefficients by applying shifting window CCA
        highest_correlation, detected_choice = self.evaluate_correlation_coefficients(self.correlation_coefficients) # evaluating the correlation coefficients to get the highest correlation and the detected choice
        self.decoded_choices.append[detected_choice]
    def apply_shifting_window_cca(self, band_data):
        """
        Applies shifting window CCA to the filtered band data.
        """
        correlation_coefficients = {1: [], 2: [], 3: []}
        window_size = 300  # For example, 1 second window for 300 Hz sampling rate
        step_size = 150  # For example, 0.5 second step size for 300 Hz sampling rate

        for seq in correlation_coefficients.keys():
            for band_idx, band in enumerate(self.freq_bands):
                band_key = f'band_{band[0]}_{band[1]}'
                if seq in self.cca_models and band_key in self.cca_models[seq]:
                    cca_model = self.cca_models[seq][band_key]
                    num_windows = (band_data[band_key].shape[1] - window_size) // step_size + 1

                    for i in range(num_windows):
                        start = i * step_size
                        end = start + window_size
                        window_data = band_data[band_key][:, start:end]

                        # Apply the CCA model to the windowed data
                        filtered_data = cca_model.transform(window_data.T)

                        # Calculate the correlation with the corresponding frequency band template
                        correlation = self.calculate_correlations(filtered_data, seq, band_key)

                        correlation_coefficients[seq].append(correlation)

        return correlation_coefficients

    def calculate_correlations(self, filtered_data, seq, band_key):
        # Get the corresponding template for the sequence and frequency band
        template = self.templates[seq][band_key]

        # Calculate the correlation between the filtered data and the corresponding filtered template
        corr = np.corrcoef(filtered_data[:, 0], template[:, 0])[0, 1]

        return corr

    def evaluate_correlation_coefficients(self, correlation_coefficients):
        avg_correlations = {
            1: np.mean(correlation_coefficients[1]),
            2: np.mean(correlation_coefficients[2]),
            3: np.mean(correlation_coefficients[3])
        }

        # Sort the sequences by their average correlation in descending order
        sorted_correlations = sorted(avg_correlations.items(), key=lambda item: item[1], reverse=True)

        # Get the highest and second-highest correlations
        highest_sequence, highest_corr = sorted_correlations[0]
        second_highest_sequence, second_highest_corr = sorted_correlations[1]

        # Check if the highest correlation is at least 0.15 higher than the second highest
        if highest_corr >= second_highest_corr + 0.15:
            return highest_corr, highest_sequence
        else:
            return highest_corr, -1