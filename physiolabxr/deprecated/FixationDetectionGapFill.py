from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.eyetracking import gap_fill


class FixationDetection(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.max_gap_time = 0.075  # the maximum time for gap to be considered a glitch that will be filled, gap longer than this will be ignored for they are likely to be blinks

        self.gaze_channels = ['x', 'y', 'z', 'status']  # the channels of the gaze stream, x, y, z are the 3D gaze vector, status is the validity of the gaze sample
        self.gaze_status = {'valid': 2, 'invalid': 0}  # the status of the gaze sample, 2 is valid, 0 is invalid

        self.gaze_stream_name = 'Example-Eyetracking'  # the name of the gaze stream

    # loop is called <Run Frequency> times per second
    def loop(self):
        # gap filling
        # if the last sample is valid, we go back and see if there's any gap needs to be filled
        # once the gaps are filled we send the gap-filled data and clear 'em from the buffer
        if self.gaze_stream_name in self.inputs.keys():  # first check if the gaze stream is available
            gaze_timestamps = self.inputs[self.gaze_stream_name][1]  # we the gaze stream using key self.gaze_stream_name, the first element of the value is the data, the second element is the timestamps
            gaze_status = self.inputs[self.gaze_stream_name][0][self.gaze_channels.index('status')]  # we get the status channel of the gaze stream
            gaze_xyz = self.inputs[self.gaze_stream_name][0][:3]  # get the xyz channels of the gaze stream

            if gaze_status[-1] == self.gaze_status['valid']:  # and starting_invalid_duration > self.max_gap_time:  # if the sequence starts out invalid, we must wait until the end of the invalid
                gap_filled_xyz = gap_fill(gaze_xyz, gaze_status, self.gaze_status['valid'], gaze_timestamps, max_gap_time=self.max_gap_time)  # fill the gaps!
                self.outputs['gap_filled_xyz'] = gap_filled_xyz  # send the gap-filled data so we can see it in the plotter
                self.inputs.clear_stream_buffer(self.gaze_stream_name)  # clear the buffer of the gaze stream as the gaps are filled, we don't need to process them again

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

