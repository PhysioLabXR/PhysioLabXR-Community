import numpy as np

from rena.scripting.RenaScript import RenaScript
from rena.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, \
    get_baselined_event_locked_data


class ERPExtraction(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.srate = 128
        self.events = (1, 2,3 )
        self.tmin = -0.1  #
        self.tmax = 0.8
        self.baseline_time = 0.1  # time period since the erp epoch start to use as baseline
        self.erp_length = int((self.tmax - self.tmin) * 128)
        self.event_locked_data_buffer = {}
        self.eeg_channels = ["Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz"]


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
            try:
                self.inputs.clear_up_to(last_event_time)  # clear the input buffer up to the last event time
                self.event_locked_data_buffer = buffer_event_locked_data(event_locked_data, self.event_locked_data_buffer)

                if len(event_locked_data) > 0:  # if there's new data
                    if self.params['ChannelToPlot'] in self.eeg_channels:  # check if the channel to plot chosen in the params is valid
                        channel_index = self.eeg_channels.index(self.params['ChannelToPlot'])
                        baselined_data = get_baselined_event_locked_data(self.event_locked_data_buffer, channel_index, self.baseline_time, self.srate)
                        erp_viz_data = np.zeros((self.erp_length, 2))

                        if 1 in baselined_data.keys():
                            erp_viz_data[:, 0] = np.mean(baselined_data[1], axis=0) if self.params['PlotAverage'] else baselined_data[1][-1]
                        if 2 in baselined_data.keys():
                            erp_viz_data[:, 1] = np.mean(baselined_data[2], axis=0) if self.params['PlotAverage'] else baselined_data[2][-1]
                        self.outputs['ERPs'] = np.array(erp_viz_data, dtype=np.float32)
                    else:
                        print(f"Channel {self.params['ChannelToPlot']} not found")
            except Exception as e:
                print(e)


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')


