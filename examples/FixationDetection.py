import numpy as np

from rena.scripting.RenaScript import RenaScript
from rena.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, get_baselined_event_locked_data
from rena.scripting.physio.eyetracking import gap_fill


class FixationDetection(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.max_gap_time = 0.075  # the maximum time for gap to be considered a glitch that will be filled
        self.idt_window_size = 0.175

        self.gaze_channels = ['x', 'y', 'z', 'status']
        self.gaze_status = {'valid': 2, 'invalid': 0}

        self.gaze_stream_name = 'Example-Eyetracking'

    # loop is called <Run Frequency> times per second
    def loop(self):
        # gap filling
        # if the last sample is valid, we go back and see if there's any gap needs to be filled
        # once the gaps are filled we send the gap-filled data and clear 'em from the buffer

        if self.gaze_stream_name  in self.inputs.keys():
            gaze_status = self.inputs[self.gaze_stream_name][0][self.gaze_channels.index('status')]
            gaze_timestamps = self.inputs[self.gaze_stream_name][1]
            gaze_xyz = self.inputs[self.gaze_stream_name][0][:3]

            # if gaze_status[0] == self.gaze_status['invalid']:
            #     start_invalid_end_index = np.where(gaze_status != self.gaze_status['invalid'])[0][0]
            #     starting_invalid_duration = gaze_timestamps[start_invalid_end_index] - gaze_timestamps[0]
            if gaze_status[-1] == self.gaze_status['valid']:  # and starting_invalid_duration > self.max_gap_time:  # if the sequence starts out invalid, we must wait until the end of the invalid
                gap_filled_xyz = gap_fill(gaze_xyz, gaze_status, self.gaze_status['valid'], gaze_timestamps, max_gap_time=self.max_gap_time)
                self.outputs['gap_filled_xyz'] = gap_filled_xyz
                self.inputs.clear_stream_buffer(self.gaze_stream_name)

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

