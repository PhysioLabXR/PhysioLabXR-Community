from scipy.signal import butter, filtfilt, iirnotch
from sklearn.cross_decomposition import CCA
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.rpc.decorator import rpc, async_rpc
import time
class NeuralCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]         #defining frequency bands for filter bank
        self.frequency = 300                                    #default frequency of DSI-24
        self.data = DataBuffer()                                #generating a data buffer for EEG data
        self.templates = {}
        self.ccaModel = {}                                  #creating a list to store all of the CCA models
        self.ccaResults= {}
        self.decoded_choices = []                               #creating a list to store all of the decoded choices
        self.mSequence = [
            [0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0],  #mSequence1
            [0,1,1,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0],  #mSequence2
            [1,1,1,0,1,0,0,1,1,1,0,1,0,0,0,0,0,1,1,1,0,1,0,0,1,1,1,0,1,0,0]   #mSequence3
        ]
        self.sequence_length = len(self.mSequence[0])
        self.segment_length = int(np.floor(self.sequence_length*300 * 0.033))
        self.mSequenceSignal = {
            'sequence1': self.generateMSignal(0),
            'sequence2': self.generateMSignal(1),
            'sequence3': self.generateMSignal(2)
        }
        self.seq1_data = np.array([[]])
        self.seq2_data = np.array([[]])
        self.seq3_data = np.array([[]])
        self.game_start = 0


    def loop(self):
        if self.inputs:
            unfilteredData = self.inputs['DSI24'][0][14:18, :].astype(float)
            filteredData1 = self.bandpassFilter(unfilteredData, 2,120)
            filteredData2 = self.notchFilter(filteredData1,60)


            EEG_Data = {                                            #creating a dictionary for EEG data
                'stream_name': 'EEG Data',                          #defining the stream name
                'frames': filteredData2,       #choosing the correct channels
                'timestamps': self.inputs['DSI24'][1].astype(float)           #defining the timestamps
            }
            self.data.update_buffer(EEG_Data)                       #updating the data buffer with EEG data
            Data = self.data.get_data('EEG Data')
            if Data.shape[1] > 3600:     #if the data is longer than 200 seconds then cut off beginning of data so that it is to 200 seconds
                self.data.clear_stream_up_to_index(stream_name= 'EEG Data', cut_to_index= self.data.get_data('EEG Data').shape[1]-1800)
            if self.game_start == 1:                           #if training is complete (i.e there are 3 CCA models) then we can start decoding everything asyncronously
                if Data.shape[1] > self.segment_length:
                    self.decode_choice()
                if len(self.decoded_choices)> 7:
                    self.decoded_choices = self.decoded_choices[len(self.decoded_choices)-5:]


    def cleanup(self):
        self.freq_bands = [(8, 60), (12, 60), (30, 60)]
        self.mSequence = []
        self.frequency = 300
        self.data = DataBuffer()
        self.cca_models = []
        self.decoded_choices = []

        return

    #Data Manipulation
    def bandpassFilter(self, data, lowcutoff, highcutoff):
        """
        Takes in data and applies a bandpass filter to it
        :param data: EEG data to be bandpassed
        :param low: Low pass
        :param high: High pass
        :param fs: sampling frequency
        :param order: Default is 8
        :return: A bandpassed version of the data
        """
        nq = 0.5 * self.frequency
        order = 8
        lowcutoffnorm = lowcutoff / nq
        highcutoffnorm = highcutoff / nq
        b, a = butter(order, [ lowcutoffnorm, highcutoffnorm], btype = 'band')
        BPdata = filtfilt(b,a,data)
        return BPdata
    def notchFilter(self, data, notchfreq, qualityFactor = 30):
        """

        :param data: signal to be filtered
        :param notchfreq: frequency to notch
        :param qualityFactor: quality factor of the notch filter that determines the width of the notch
        :return: filtered array
        """
    # Design the notch filter
        b, a = iirnotch(notchfreq, qualityFactor, self.frequency)
        # Apply the filter to the data
        filtered_data = filtfilt(b, a, data)
        return filtered_data
    def applyFilterBank(self, data):
        """
        Returns data in 3 different arrays of different frequncy ranges
        :param data: list of data to be filtered after data has been segmented
        :return: a dictionary that contains 3 arrays for each frequency band where the keys are the found in self.freq_bands
        """
        band = {}           #Dictionary created to fill in
        for i in range(3):
            filtered_segments = np.empty((4,data[0].shape[1],0))  # List to hold filtered segments for the current frequency band
            for segment in data:
                # Apply bandpass filter to each segment and append the result to the list
                filtered_segment = self.bandpassFilter(segment, self.freq_bands[i][0], self.freq_bands[i][1])
                filtered_segments = np.concatenate((filtered_segments, filtered_segment[:,:,np.newaxis]),axis =2)

            # Average the filtered segments across the first axis
            band[self.freq_bands[i]] = np.mean(filtered_segments, axis=2)
        return band
    def adjust_segments(self, segments):
        adjusted_segments = []
        for segment in segments:
            # If the segment is shorter than the desired length, pad it with zeros
            if segment.shape[1] < self.segment_length:
                padding = np.zeros((segment.shape[0], self.segment_length - segment.shape[1]))
                adjusted_segment = np.hstack((segment, padding))  # Pad with zeros
            else:
                # If the segment is longer, trim it to the desired length
                adjusted_segment = segment[:, :self.segment_length]

            adjusted_segments.append(adjusted_segment)

        return adjusted_segments
    def createTemplates(self):
        """
        Creates templates that are EEG arrays that are filtered
        :param data: EEG data for segmenting into templates
        :return: dictionary of dictionary of filtered EEG arrays Keys: segment number -> keys frequency band
        """
        seq1_segments = np.array_split(self.seq1_data, self.seq1_data.shape[1] // self.segment_length, axis=1)  #size segment_length
        seq2_segments = np.array_split(self.seq2_data, self.seq2_data.shape[1] // self.segment_length, axis=1)
        seq3_segments = np.array_split(self.seq3_data, self.seq3_data.shape[1] // self.segment_length, axis=1)

        seq1_segments = self.adjust_segments(seq1_segments) #size segment_length
        seq2_segments = self.adjust_segments(seq2_segments)
        seq3_segments = self.adjust_segments(seq3_segments)


        self.templates['sequence1'] = self.applyFilterBank(seq1_segments) #size_segment_length
        self.templates['sequence2'] = self.applyFilterBank(seq2_segments)
        self.templates['sequence3'] = self.applyFilterBank(seq3_segments)

    def generateMSignal(self, seqNum):

        samples_per_bit = self.segment_length // len(self.mSequence[seqNum])
        signal = np.repeat(self.mSequence[seqNum], samples_per_bit )

        ##Change signal to be another dictionary holding freq bands and the respective

        # Step 4: If the signal is longer than required, truncate it
        if len(signal) > self.segment_length:
            signal = signal[:, :self.segment_length]


        elif len(signal) < self.segment_length:
            padding = int(np.ceil((self.segment_length - len(signal))/2)+1)
            signal = np.pad(signal, pad_width=padding, mode = 'constant', constant_values= 0)
            signal = signal[:self.segment_length]
        Msignal = [np.tile(signal, (4, 1))]
        FilteredMsignal = self.applyFilterBank(Msignal)
        return FilteredMsignal

    def train_cca(self):
        """
        Training the CCA model
        By generated spatial filters and templates for each target m-sequence
        """
        self.createTemplates()
        self.ccaModel = {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)}
        for sequence in self.templates.keys():
            for freqBand in self.templates[sequence].keys():
                cca = CCA(n_components = 1)
                cca.fit(self.templates[sequence][freqBand].T, self.mSequenceSignal[sequence][freqBand].T)
                self.ccaModel[sequence][freqBand] = cca
        print('training done')

    @async_rpc
    def add_seq_data(self, sequenceNum: int, duration: float):  # Data is going to come in sequencially seq1 -> seq2 -> seq3 repeat
        eegData = self.data.get_data('EEG Data')[:, int(-duration * 300):]  # 4 by (samples)

        if sequenceNum == 1:
            if self.seq1_data.size == 0:
                self.seq1_data = eegData
            else:
                self.seq1_data = np.concatenate((self.seq1_data, eegData), axis=1)
        elif sequenceNum == 2:
            if self.seq2_data.size == 0:
                self.seq2_data = eegData
            else:
                self.seq2_data = np.concatenate((self.seq2_data, eegData), axis=1)
        elif sequenceNum == 3:
            if self.seq3_data.size == 0:
                self.seq3_data = eegData
            else:
                self.seq3_data = np.concatenate((self.seq3_data, eegData), axis=1)

    @async_rpc
    def training(self) -> int:
        """
        Args:
            input0: int - 1 for choice 1, 2 for choice 2, 3 for choice 3
        Returns: Generates correlation coefficients for EEG data x m-sequence
        """
        # Train the CCA
        self.train_cca()  # start training the CCA
        self.data.clear_buffer_data() #Clear the buffer after training
        return 1
    @async_rpc
    def decode(self) -> int:
        self.game_start = 1
        # Get the choices decoded so far
        choices = self.decoded_choices
        if len([x for x in choices if x != None]) > 2:
            # Determine the most common choice
            user_choice = max(set([x for x in choices if x != None]), key=choices.count)
            print(choices, user_choice)
            self.decoded_choices = []
            self.data.clear_buffer_data()


        else:
            user_choice = 0
        # Return the most common detected choice
        return user_choice
    def decode_choice(self):

        self.correlation_coefficients = self.apply_shifting_window_cca(self.data.get_data('EEG Data'))  # getting the correlation coefficients by applying shifting window CCA
        highest_correlation, detected_choice = self.evaluate_correlation_coefficients(self.correlation_coefficients)  # evaluating the correlation coefficients to get the highest correlation and the detected choice
        if detected_choice != None:
            self.decoded_choices.append(detected_choice)

    def apply_shifting_window_cca(self, data):
        """
        Applies shifting window CCA to the filtered band data.
        """
        window_size = int(self.sequence_length * 0.033 * 300)  # For example, 1 second window for 300 Hz sampling rate
        step_size = int(window_size/4)  # For example, 0.5 second step size for 300 Hz sampling rate

        segments = []
        for start in range(0, data.shape[1], step_size):
            if start+window_size < data.shape[1]:
                segment = data[:, start:start+window_size]
                segments.append(segment) #For some reason the segments created are not of the shape

        #Filter the data
        filtered_data = self.applyFilterBank(segments)
        self.ccaResults =  {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)} #temporarily assigns the dictionary so that we can use the keys

        correlation = {k: {sub_k: None for sub_k in v.keys()} for k, v in self.templates.items() if isinstance(v, dict)}
        avg_correlation = {}
        #Transform the data with CCA
        for sequence in self.templates.keys():
            for freqBand in self.templates[sequence].keys():
                cca = self.ccaModel[sequence][freqBand]
                self.ccaResults[sequence][freqBand], refMSeq = cca.transform(filtered_data[freqBand].T, self.mSequenceSignal[sequence][freqBand].T)
                correlation[sequence][freqBand] = np.abs(np.corrcoef(self.ccaResults[sequence][freqBand].flatten(), refMSeq.T)[0, 1])
                print(correlation[sequence][freqBand], freqBand, sequence)
            avg_correlation[sequence] = np.mean(list(correlation[sequence].values()))
        return avg_correlation

    def evaluate_correlation_coefficients(self, correlation_coefficients):
        # Sort the sequences by their average correlation in descending order
        sorted_correlations = sorted(correlation_coefficients.items(), key=lambda item: item[1], reverse=True)
        # Get the highest and second-highest correlations
        highest_sequence, highest_corr = sorted_correlations[0]
        second_highest_sequence, second_highest_corr = sorted_correlations[1]
        # Check if the highest correlation is at least 0.15 higher than the second highest
        if highest_corr >= second_highest_corr + 0.05:
            return highest_corr, int(highest_sequence[-1])
        else:
            return highest_corr, None
