import cv2
import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, get_baselined_event_locked_data
from physiolabxr.scripting.physio.eyetracking import gap_fill


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

        self.gaze_timestamp_head = 0
        self.video_stream_name = 'Example-Video'
        self.video_shape = (400, 400, 3)

        self.frame_gaze_pixel_stream_name = 'Example-Video-Gaze-Pixel'


    # loop is called <Run Frequency> times per second
    def loop(self):
        # gap filling
        # if the last sample is valid, we go back and see if there's any gap needs to be filled
        # once the gaps are filled we send the gap-filled data and clear 'em from the buffer

        if self.gaze_stream_name in self.inputs.keys():
            gaze_status = self.inputs[self.gaze_stream_name][0][self.gaze_channels.index('status')]
            gaze_timestamps = self.inputs[self.gaze_stream_name][1]
            gaze_xyz = self.inputs[self.gaze_stream_name][0][:3]

            # if gaze_status[0] == self.gaze_status['invalid']:
            #     start_invalid_end_index = np.where(gaze_status != self.gaze_status['invalid'])[0][0]
            #     starting_invalid_duration = gaze_timestamps[start_invalid_end_index] - gaze_timestamps[0]
            if gaze_status[-1] == self.gaze_status['valid']:  # and starting_invalid_duration > self.max_gap_time:  # if the sequence starts out invalid, we must wait until the end of the invalid
                gap_filled_xyz = gap_fill(gaze_xyz, gaze_status, self.gaze_status['valid'], gaze_timestamps, max_gap_time=self.max_gap_time, verbose=False)
                self.outputs['gap_filled_xyz'] = gap_filled_xyz
                self.inputs.clear_stream_buffer(self.gaze_stream_name)
                self.gaze_timestamp_head = gaze_timestamps[-1]

        # release video frames up to the processed gaze timestamp
        # but we only release one video frame per loop
        if self.video_stream_name in self.inputs.keys():
            video_timestamps = self.inputs[self.video_stream_name][1]
            video_frames = self.inputs[self.video_stream_name][0]
            frame_pixels = self.inputs[self.frame_gaze_pixel_stream_name][0]  # find the frame pixel corresponding to the video timestamp
            frame_pixel_timestamps = self.inputs[self.frame_gaze_pixel_stream_name][1]

            if video_timestamps[0] < self.gaze_timestamp_head:
                this_frame = video_frames[:, 0].reshape(self.video_shape).copy()
                this_frame_timestamp = video_timestamps[0]
                this_frame_pixel = frame_pixels[:, frame_pixel_timestamps == this_frame_timestamp]
                if len(this_frame_pixel) > 0:
                    cv2.circle(this_frame, np.array(this_frame_pixel[:, 0], dtype=np.uint8), 10, (255, 0, 0), 2)
                self.outputs['gaze_processed_video'] = this_frame.reshape(-1)
                # remove the first video frame from the buffer
                self.inputs.clear_stream_up_to_index(self.video_stream_name, 1)
                self.inputs.clear_stream_up_to(self.frame_gaze_pixel_stream_name, this_frame_timestamp)

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

