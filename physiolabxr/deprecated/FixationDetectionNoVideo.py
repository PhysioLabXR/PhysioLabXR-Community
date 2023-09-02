from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.eyetracking import gap_fill, fixation_detection_idt
from physiolabxr.utils.buffers import DataBuffer


class FixationDetection(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        self.max_gap_time = 0.075  # the maximum time for gap to be considered a glitch that will be filled

        self.gaze_channels = ['x', 'y', 'z', 'status']  # the channels of the gaze stream, x, y, z are the 3D gaze vector, status is the validity of the gaze sample
        self.gaze_status = {'valid': 2, 'invalid': 0}  # the status of the gaze sample, 2 is valid, 0 is invalid

        self.gaze_stream_name = 'Example-Eyetracking'  # the name of the gaze stream

        self.fixation_timestamp_head = 0  # the timestamp of the beginning of the last fixation window

        self.processed_gaze_buffer = DataBuffer(stream_buffer_sizes={'fixations': 1000, 'gap_filled_xyz': 1000})  # buffer to store the preprocessed gaze data, including the gap-filled gaze vectors and the fixation sequences


    # loop is called <Run Frequency> times per second
    def loop(self):
        # gap filling
        # if the last sample is valid, we go back and see if there's any gap needs to be filled
        # once the gaps are filled we send the gap-filled data and clear 'em from the buffer
        if self.gaze_stream_name in self.inputs.keys():  # first check if the gaze stream is available
            gaze_status = self.inputs[self.gaze_stream_name][0][self.gaze_channels.index('status')]  # we the gaze stream using key self.gaze_stream_name, the first element of the value is the data, the second element is the timestamps
            gaze_timestamps = self.inputs[self.gaze_stream_name][1]  # we get the status channel of the gaze stream
            gaze_xyz = self.inputs[self.gaze_stream_name][0][:3]  # get the xyz channels of the gaze stream

            if gaze_status[-1] == self.gaze_status['valid']:  # if the sequence starts out invalid, we must wait until the end of the invalid
                gap_filled_xyz = gap_fill(gaze_xyz, gaze_status, self.gaze_status['valid'], gaze_timestamps, max_gap_time=self.max_gap_time, verbose=False)  # fill the gaps!
                self.processed_gaze_buffer.update_buffer({'stream_name': 'gap_filled_xyz', 'frames': gap_filled_xyz, 'timestamps': gaze_timestamps})  # add the gap filled data to the buffer, so we can use it for fixation detection
                self.outputs['gap_filled_xyz'] = gap_filled_xyz  # send the gap-filled data so we can see it in the plotter
                self.inputs.clear_stream_buffer(self.gaze_stream_name)  # clear the gaze stream, so we don't process the same data again, the fixation detection will act on the gap filled data

            # up to the point of the last gap filled index, we detect fixation. The idt window ends at the gap filled index
            fixations, last_window_start = fixation_detection_idt(*self.processed_gaze_buffer['gap_filled_xyz'], window_size=self.params['idt_window_size'], dispersion_threshold_degree=self.params['dispersion_threshold_degree'], return_last_window_start=True)
            self.outputs['fixations'] = fixations[0:1]  # send the fixations, we grab the first column of the result, the second column are the timestamps
            self.processed_gaze_buffer.clear_stream_up_to_index('gap_filled_xyz', last_window_start)  # now clear the gap filled data up to the last window start, so we don't process the same data again

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

