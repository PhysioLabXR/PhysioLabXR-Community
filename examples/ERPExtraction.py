import numpy as np

from rena.scripting.RenaScript import RenaScript
from rena.scripting.physio.epochs import get_event_locked_data


class ERPExtraction(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.events = (1, 2,3 )
        self.tmin = -0.1  #
        self.tmax = 0.8
        self.baseline_time = 0.1  # time period since the erp epoch start to use as baseline
        self.erp_length = int((self.tmax - self.tmin) * 128)
        self.targets = np.empty((0, 9, self.erp_length))
        self.distractors = np.empty((0, 9, self.erp_length))
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
                                                                       return_last_event_time=True)
            try:
                self.inputs.clear_up_to(last_event_time)  # clear the input buffer up to the last event time
                if len(event_locked_data) > 0:
                    if self.params['ChannelToPlot'] in self.eeg_channels:
                        # create the data for visualization
                        if 1 in event_locked_data.keys():
                            self.distractors = np.concatenate([self.distractors, event_locked_data[1]])
                            # apply baseline correction
                            n_trials = event_locked_data[1].shape[0]
                            print(f"Found {n_trials} new distractor event, now has {self.distractors.shape[0]} trials for distractors")
                        if 2 in event_locked_data.keys():
                            self.targets = np.concatenate([self.targets, event_locked_data[2]])
                            n_trials = event_locked_data[2].shape[0]
                            print(f"Found {n_trials} target event, now has {self.targets.shape[0]} trials for targets")
                        # if any data is available, send it to the output
                        if 1 in event_locked_data.keys() or 2 in event_locked_data.keys():  # only send data if there is new data
                            erp_viz_data = np.zeros((self.erp_length, 2))
                            if self.distractors.shape[0] > 0:
                                distractor_data = self.distractors - np.mean(self.distractors[:, :, :int(self.baseline_time * 128)], axis=2, keepdims=True)
                                distractor_data = distractor_data[:, self.eeg_channels.index(self.params['ChannelToPlot'])]
                                distractor_data = np.mean(distractor_data, axis=0)  # average across trials
                                erp_viz_data[:, 0] = distractor_data
                            if self.targets.shape[0] > 0:
                                target_data = self.targets - np.mean(self.targets[:, :, :int(self.baseline_time * 128)], axis=2, keepdims=True)
                                target_data = target_data[:, self.eeg_channels.index(self.params['ChannelToPlot'])]
                                target_data = np.mean(target_data, axis=0)  # average across trials
                                erp_viz_data[:, 1] = target_data
                            self.outputs['ERPs'] = np.array(erp_viz_data, dtype=np.float32)
                    else:
                        print(f"Channel {self.params['ChannelToPlot']} not found")
            except Exception as e:
                print(e)


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')


