from collections import deque

import numpy as np

from physiolabxr.utils.realtime_DSP import DataProcessor


class GazeFilterFixationDetectionIDTAngular(DataProcessor):
    def __init__(self,
                 sampling_frequency=250, duration=100,
                 sampling_frequency_unit_duration_unit_scaling_factor=1000,
                 angular_threshold_degree=1.2, dtype=np.float64):
        super().__init__()

        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.sampling_frequency_unit_duration_unit_scaling_factor = sampling_frequency_unit_duration_unit_scaling_factor
        self.angular_threshold_degree = angular_threshold_degree
        self.dtype = dtype

        self._gaze_data_buffer = None
        self.buffer_size = None
        self._gaze_vector_buffer = None

    def evoke_function(self):
        self.buffer_size = int(self.sampling_frequency * (
                self.duration / self.sampling_frequency_unit_duration_unit_scaling_factor))

        self._gaze_data_buffer = deque(maxlen=int(self.buffer_size))
        self._gaze_vector_buffer = deque(maxlen=int(self.buffer_size))

    def set_data_processor_params(self, sampling_frequency=250, duration=150,
                                  sampling_frequency_unit_duration_unit_scaling_factor=1000,
                                  angular_threshold_degree=1.5, dtype=np.float64):
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.sampling_frequency_unit_duration_unit_scaling_factor = sampling_frequency_unit_duration_unit_scaling_factor
        self.angular_threshold_degree = angular_threshold_degree
        self.dtype = dtype

    def process_sample(self, gaze_data: GazeData):
        # centroid of the gaze data will be assigned
        self._gaze_data_buffer.appendleft(gaze_data)
        # self._gaze_vector_buffer.appendleft(gaze_data.combined_eye_gaze_data.gaze_direction)

        if gaze_data.combined_eye_gaze_data.gaze_point_valid:
            self._gaze_vector_buffer.appendleft(gaze_data.combined_eye_gaze_data.gaze_direction)
        else:
            # empty the buffer
            self._gaze_vector_buffer.clear()
            self._gaze_data_buffer.appendleft(gaze_data)

        if len(self._gaze_vector_buffer) == self.buffer_size:
            angular_dispersion = calculate_angular_dispersion(self._gaze_vector_buffer, self.dtype)
            if angular_dispersion <= self.angular_threshold_degree:
                gaze_data.gaze_type = GazeType.FIXATION
            else:
                gaze_data.gaze_type = GazeType.SACCADE
        else:
            gaze_data.gaze_type = GazeType.UNDETERMINED

        return gaze_data