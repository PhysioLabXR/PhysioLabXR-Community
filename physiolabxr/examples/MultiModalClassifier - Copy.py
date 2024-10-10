import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, get_baselined_event_locked_data


class ERPExtraction(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.events = (1, 2, 3)  # 1 is distractor, 2 is target, 3 is novelty
        self.tmin = -0.1  # Time before event marker to include in the epoch
        self.tmax = 0.8  # Time after event marker to include in the epoch
        self.baseline_time = 0.1  # Time period since the ERP epoch start to use as baseline
        self.erp_length = int((self.tmax - self.tmin) * 128)  # Length of the ERP epoch in samples
        self.event_locked_data_buffer = {}  # Dictionary to store event-locked data
        self.eeg_channels = self.get_stream_info('Example-BioSemi-Midline', 'ChannelNames')  # List of EEG channels
        self.srate = self.get_stream_info('Example-BioSemi-Midline', 'NominalSamplingRate')  # Sampling rate of the EEG data in Hz


    # loop is called <Run Frequency> times per second
    def loop(self):
        # first check if the inputs are available
        if 'Example-EventMarker' in self.inputs.keys() and 'Example-BioSemi-Midline' in self.inputs.keys():
            event_locked_data, last_event_time = get_event_locked_data(event_marker=self.inputs['Example-EventMarker'],
                                                                       data=self.inputs['Example-BioSemi-Midline'],
                                                                       events_of_interest=self.events,
                                                                       tmin=self.tmin,
                                                                       tmax=self.tmax,
                                                                       srate=128,
                                                                       return_last_event_time=True, verbose=1)
            self.inputs.clear_up_to(last_event_time)  # Clear the input buffer up to the last event time to avoid processing duplicate data
            self.event_locked_data_buffer = buffer_event_locked_data(event_locked_data, self.event_locked_data_buffer)  # Buffer the event-locked data for further processing

            if len(event_locked_data) > 0:  # if there's new data
                if self.params['ChannelToPlot'] in self.eeg_channels:  # check if the channel to plot chosen in the params is valid
                    channel_index = self.eeg_channels.index(self.params['ChannelToPlot'])  # Get the index of the chosen EEG channel from the list
                    baselined_data = get_baselined_event_locked_data(self.event_locked_data_buffer, channel_index, self.baseline_time, self.srate)  # Obtain baselined event-locked data for the chosen channel
                    erp_viz_data = np.zeros((self.erp_length, 2))  # Create a visualization data array for ERP

                    # Populate the visualization data with ERP values from different events (if available)
                    if 1 in baselined_data.keys():
                        erp_viz_data[:, 0] = np.mean(baselined_data[1], axis=0) if self.params['PlotAverage'] else baselined_data[1][-1]
                    if 2 in baselined_data.keys():
                        erp_viz_data[:, 1] = np.mean(baselined_data[2], axis=0) if self.params['PlotAverage'] else baselined_data[2][-1]
                    self.outputs['ERPs'] = np.array(erp_viz_data, dtype=np.float32)  # Set the output 'ERPs' with the visualization data
                else:
                    print(f"Channel {self.params['ChannelToPlot']} not found")

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')


