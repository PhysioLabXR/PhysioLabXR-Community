import cv2
import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.eyetracking import gap_fill, fixation_detection_idt
from physiolabxr.scripting.physio.utils import time_to_index
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
        self.frame_gaze_pixel_stream_name = 'Example-Video-Gaze-Pixel'  # the name of the frame gaze pixel stream, the stream tells us where the gaze is on the 400x400 video frame

        self.video_stream_name = 'Example-Video'  # the name of the video stream
        self.video_shape = (400, 400, 3)  # the shape of the video stream, because all inputs are flattened as they comes in, we need to reshape the video frames to be able to put shapes on them
        self.fixation_circle_color = (255, 0, 0)  # when the video frame's timestamp matches a fixation's, we put a red circle at the pixel location of where the gaze is
        self.gaze_circle_color = (0, 0, 255)  # the there's no fixation, we put a blue circle at the gaze location


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
            self.processed_gaze_buffer.update_buffer({'stream_name': 'fixations', 'frames': fixations[0:1], 'timestamps': fixations[1]})  # add the gap filled data to the buffer, so we can use it for fixation detection
            self.outputs['fixations'] = fixations[0:1]  # send the fixations, we grab the first column of the result, the second column are the timestamps
            self.fixation_timestamp_head = self.processed_gaze_buffer['gap_filled_xyz'][1][last_window_start]  # update the gaze timestamp head, so we can release video frames up to this timestamp
            self.processed_gaze_buffer.clear_stream_up_to_index('gap_filled_xyz', last_window_start)  # now clear the gap filled data up to the last window start, so we don't process the same data again

        # release video frames up to the processed gaze timestamp, but we only release one video frame per loop
        # we loop through the video frames, if the timestamp of the video frame is less than the timestamp of the last fixation, we release the video frame and remove it from the buffer
        # we keep doing this until the timestamp of the video frame is greater than the timestamp of the last fixation or there's no more video frames
        while self.video_stream_name in self.inputs.keys() and len(self.inputs[self.video_stream_name][1]) > 0 and self.inputs[self.video_stream_name][1][0] < self.fixation_timestamp_head:
            video_frames = self.inputs[self.video_stream_name][0]  # get the video frames
            frame_pixels = self.inputs[self.frame_gaze_pixel_stream_name][0]  # find the frame pixel corresponding to the video timestamp
            frame_pixel_timestamps = self.inputs[self.frame_gaze_pixel_stream_name][1]  # get the timestamps of the frame pixels

            # we first look at the first frame in the buffer, we already know it's timestamp is less than the timestamp of the last fixation from the while condition
            this_frame = video_frames[:, 0].reshape(self.video_shape).copy()  # take the first frame in the buffer, make a copy so the data is contiguous
            this_frame_timestamp = self.inputs[self.video_stream_name][1][0]  # get the timestamp of the first frame in the buffer
            this_frame_pixel = frame_pixels[:, frame_pixel_timestamps == this_frame_timestamp]  # get where the participant is look at in pixel coordinates

            # find the closest fixation to the current video frame, we need to call time_to_index because the timestamps of the fixations are not the same as the timestamps of the video frames
            # this is different from the frame pixel in the line above, whose timestamps are the same as the video frames and we can find exact matches
            fixation_index = time_to_index(self.processed_gaze_buffer['fixations'][1], this_frame_timestamp)  # find the index of the closest fixation to the current video frame
            is_fixation = self.processed_gaze_buffer['fixations'][0][:, fixation_index][0]  # find the fixation value
            color = self.fixation_circle_color if is_fixation else self.gaze_circle_color  # if the participant is fixating, we draw a red circle, otherwise we draw a green circle
            if this_frame_pixel.shape[1] > 0:  # if we can find a matching gaze coordinate, then we draw a circle on the video frame
                cv2.circle(this_frame, np.array(this_frame_pixel[:, 0], dtype=np.uint8), 10, color, 2)  # draw a circle on the video frame
            self.outputs['gaze_processed_video'] = this_frame.reshape(-1)  # send the video frame to the plotter
            self.inputs.clear_stream_up_to_index(self.video_stream_name, 1)  # remove the first video frame from the buffer
            self.processed_gaze_buffer.clear_stream_up_to_index('fixations', fixation_index)  # also remove the fixation up to the video frame we just released, we don't need it anymore
            self.inputs.clear_stream_up_to(self.frame_gaze_pixel_stream_name, this_frame_timestamp)  # do the same for frame pixel, but we use the timestamp of the video frame

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

