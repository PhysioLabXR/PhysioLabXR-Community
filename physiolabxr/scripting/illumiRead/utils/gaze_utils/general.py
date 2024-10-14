import enum
import numpy as np
from physiolabxr.utils.dsp_utils.dsp_modules import DataProcessor


class GazeType(enum.Enum):
    SACCADE = 1
    FIXATION = 2
    UNDETERMINED = 0


def angle_between_vectors(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def angular_velocity_between_vectors_radians(v1, v2, time_delta):
    """Returns the angular velocity in radians per second between vectors 'v1' and 'v2'"""
    angle = angle_between_vectors(v1, v2)
    return angle / time_delta


def angular_velocity_between_vectors_degrees(v1, v2, time_delta):
    """Returns the angular velocity in radians per second between vectors 'v1' and 'v2'"""
    angle = angle_between_vectors(v1, v2)
    return np.degrees(angle) / time_delta


class GazeData:
    def __init__(self):
        pass

    def get_combined_eye_gaze_data_valid(self):
        return False

    def get_combined_eye_gaze_direction(self):
        return np.zeros(3)

    def get_timestamp(self):
        return 0

    def set_gaze_type(self, gaze_type):
        pass

    def get_gaze_type(self):
        return GazeType.UNDETERMINED




class GazeFilterFixationDetectionIVT(DataProcessor):
    def __init__(self, angular_speed_threshold_degree=100):
        super().__init__()
        self.last_gaze_data = GazeData()
        self.angular_threshold_degree = angular_speed_threshold_degree
        self.invalid_gaze_data_count = 0

    def process_sample(self, gaze_data: GazeData):
        if self.last_gaze_data.get_combined_eye_gaze_data_valid() and gaze_data.get_combined_eye_gaze_data_valid():  # if both gaze data valid

            speed = angular_velocity_between_vectors_degrees(
                self.last_gaze_data.get_combined_eye_gaze_direction(),
                gaze_data.get_combined_eye_gaze_direction(),
                time_delta=gaze_data.get_timestamp() - self.last_gaze_data.get_timestamp())
            # print(speed)
            if speed <= self.angular_threshold_degree:
                gaze_data.set_gaze_type(GazeType.FIXATION)
            else:
                gaze_data.set_gaze_type(GazeType.SACCADE)
        else:
            pass
            # self.invalid_gaze_data_count += 1
            # print('invalid gaze data:', self.invalid_gaze_data_count)
        self.last_gaze_data = gaze_data

        return gaze_data

    def reset_data_processor(self):
        self.last_gaze_data = GazeData()
