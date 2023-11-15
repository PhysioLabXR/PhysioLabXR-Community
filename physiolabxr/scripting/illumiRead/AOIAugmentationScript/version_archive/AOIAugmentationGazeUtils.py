import enum
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import TobiiProFusionChannel
from physiolabxr.utils.dsp_utils.dsp_modules import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

class GazeType(enum.Enum):
    SACCADE = 1
    FIXATION = 2
    UNDETERMINED = 0


class EyeData:
    def __init__(self, gaze_origin_in_user_coordinate=np.array([0, 0, 0]),
                 gaze_point_in_user_coordinate=np.array([0, 0, 0]), gaze_origin_valid=False,
                 gaze_point_valid=False,
                 gaze_origin_in_trackbox_coordinate=np.array([0, 0, 0]),
                 pupil_diameter=0.0,
                 pupil_diameter_valid=False,
                 gaze_point_on_display_area=np.array([0, 0]),
                 timestamp=0):

        self.gaze_origin_in_user_coordinate = gaze_origin_in_user_coordinate
        self.gaze_point_in_user_coordinate = gaze_point_in_user_coordinate
        self.gaze_origin_valid = gaze_origin_valid
        self.gaze_point_valid = gaze_point_valid
        self.gaze_origin_in_trackbox_coordinate = gaze_origin_in_trackbox_coordinate
        self.pupil_diameter = pupil_diameter
        self.pupil_diameter_valid = pupil_diameter_valid
        self.gaze_point_on_display_area = gaze_point_on_display_area
        self.timestamp = timestamp

        self.gaze_direction = np.array([0, 0, 0])

        if self.gaze_origin_valid and self.gaze_point_valid:
            self.gaze_direction = self.get_gaze_direction()

    def get_gaze_direction(self, normalize=True):
        gaze_direction = self.gaze_point_in_user_coordinate - self.gaze_origin_in_user_coordinate
        if normalize:
            gaze_direction = gaze_direction / np.linalg.norm(gaze_direction)
        return gaze_direction


class GazeData:

    def __init__(self, left_eye_gaze_data: EyeData = EyeData(), right_eye_gaze_data: EyeData = EyeData()):
        self.left_eye_gaze_data = left_eye_gaze_data
        self.right_eye_gaze_data = right_eye_gaze_data
        self.combined_eye_gaze_data = EyeData()
        self.get_combined_eye_gaze_data()
        self.timestamp = self.combined_eye_gaze_data.timestamp

        self.gaze_type = GazeType.UNDETERMINED

    def get_combined_eye_gaze_data(self):
        self.combined_eye_gaze_data = EyeData(
            gaze_origin_in_user_coordinate=(self.left_eye_gaze_data.gaze_origin_in_user_coordinate + self.right_eye_gaze_data.gaze_origin_in_user_coordinate) / 2,
            gaze_point_in_user_coordinate=(self.left_eye_gaze_data.gaze_point_in_user_coordinate + self.right_eye_gaze_data.gaze_point_in_user_coordinate) / 2,
            gaze_origin_valid=self.left_eye_gaze_data.gaze_origin_valid and self.right_eye_gaze_data.gaze_origin_valid,
            gaze_point_valid=self.left_eye_gaze_data.gaze_point_valid and self.right_eye_gaze_data.gaze_point_valid,
            gaze_origin_in_trackbox_coordinate=(self.left_eye_gaze_data.gaze_origin_in_trackbox_coordinate + self.right_eye_gaze_data.gaze_origin_in_trackbox_coordinate) / 2,
            pupil_diameter=(self.left_eye_gaze_data.pupil_diameter + self.right_eye_gaze_data.pupil_diameter) / 2,
            pupil_diameter_valid=self.left_eye_gaze_data.pupil_diameter_valid and self.right_eye_gaze_data.pupil_diameter_valid,
            gaze_point_on_display_area= (self.left_eye_gaze_data.gaze_point_on_display_area + self.right_eye_gaze_data.gaze_point_on_display_area) / 2,
            timestamp=self.left_eye_gaze_data.timestamp
        )
        self.timestamp = self.combined_eye_gaze_data.timestamp/1000000

    def construct_gaze_data_tobii_pro_fusion(self, gaze_data_t):
        left_gaze_origin_in_user_coordinate = gaze_data_t[[TobiiProFusionChannel.LeftGazeOriginInUserCoordinatesX,
                                                           TobiiProFusionChannel.LeftGazeOriginInUserCoordinatesY,
                                                           TobiiProFusionChannel.LeftGazeOriginInUserCoordinatesZ]]

        left_gaze_point_in_user_coordinate = gaze_data_t[[TobiiProFusionChannel.LeftGazePointInUserCoordinatesX,
                                                          TobiiProFusionChannel.LeftGazePointInUserCoordinatesY,
                                                          TobiiProFusionChannel.LeftGazePointInUserCoordinatesZ]]

        left_gaze_origin_in_track_box_coordinate = gaze_data_t[
            [TobiiProFusionChannel.LeftGazeOriginInTrackBoxCoordinatesX,
             TobiiProFusionChannel.LeftGazeOriginInTrackBoxCoordinatesY,
             TobiiProFusionChannel.LeftGazeOriginInTrackBoxCoordinatesZ]]

        left_gaze_origin_valid = gaze_data_t[TobiiProFusionChannel.LeftGazeOriginValid]
        left_gaze_point_valid = gaze_data_t[TobiiProFusionChannel.LeftGazePointValid]

        left_pupil_diameter = gaze_data_t[TobiiProFusionChannel.LeftPupilDiameter]
        left_pupil_diameter_valid = gaze_data_t[TobiiProFusionChannel.LeftPupilDiameterValid]

        left_gaze_point_on_display_area = gaze_data_t[
            [TobiiProFusionChannel.LeftGazePointOnDisplayAreaX, TobiiProFusionChannel.LeftGazePointOnDisplayAreaY]]

        self.left_eye_gaze_data = EyeData(
            gaze_origin_in_user_coordinate=left_gaze_origin_in_user_coordinate,
            gaze_point_in_user_coordinate=left_gaze_point_in_user_coordinate,
            gaze_origin_valid=left_gaze_origin_valid,
            gaze_point_valid=left_gaze_point_valid,
            gaze_origin_in_trackbox_coordinate=left_gaze_origin_in_track_box_coordinate,
            pupil_diameter=left_pupil_diameter,
            pupil_diameter_valid=left_pupil_diameter_valid,
            gaze_point_on_display_area=left_gaze_point_on_display_area,
            timestamp=gaze_data_t[TobiiProFusionChannel.OriginalGazeDeviceTimeStamp])

        right_gaze_origin_in_user_coordinate = gaze_data_t[[TobiiProFusionChannel.RightGazeOriginInUserCoordinatesX,
                                                            TobiiProFusionChannel.RightGazeOriginInUserCoordinatesY,
                                                            TobiiProFusionChannel.RightGazeOriginInUserCoordinatesZ]]

        right_gaze_point_in_user_coordinate = gaze_data_t[[TobiiProFusionChannel.RightGazePointInUserCoordinatesX,
                                                           TobiiProFusionChannel.RightGazePointInUserCoordinatesY,
                                                           TobiiProFusionChannel.RightGazePointInUserCoordinatesZ]]

        right_gaze_origin_in_track_box_coordinate = gaze_data_t[
            [TobiiProFusionChannel.RightGazeOriginInTrackBoxCoordinatesX,
             TobiiProFusionChannel.RightGazeOriginInTrackBoxCoordinatesY,
             TobiiProFusionChannel.RightGazeOriginInTrackBoxCoordinatesZ]]

        right_gaze_origin_valid = gaze_data_t[TobiiProFusionChannel.RightGazeOriginValid]
        right_gaze_point_valid = gaze_data_t[TobiiProFusionChannel.RightGazePointValid]

        right_pupil_diameter = gaze_data_t[TobiiProFusionChannel.RightPupilDiameter]
        right_pupil_diameter_valid = gaze_data_t[TobiiProFusionChannel.RightPupilDiameterValid]

        right_gaze_point_on_display_area = gaze_data_t[
            [TobiiProFusionChannel.RightGazePointOnDisplayAreaX, TobiiProFusionChannel.RightGazePointOnDisplayAreaY]]

        self.right_eye_gaze_data = EyeData(
            gaze_origin_in_user_coordinate=right_gaze_origin_in_user_coordinate,
            gaze_point_in_user_coordinate=right_gaze_point_in_user_coordinate,
            gaze_origin_valid=right_gaze_origin_valid,
            gaze_point_valid=right_gaze_point_valid,
            gaze_origin_in_trackbox_coordinate=right_gaze_origin_in_track_box_coordinate,
            pupil_diameter=right_pupil_diameter,
            pupil_diameter_valid=right_pupil_diameter_valid,
            gaze_point_on_display_area=right_gaze_point_on_display_area,
            timestamp=TobiiProFusionChannel.OriginalGazeDeviceTimeStamp)

        self.get_combined_eye_gaze_data()


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


class GazeFilterFixationDetectionIVT(DataProcessor):
    def __init__(self, angular_speed_threshold_degree=100):
        super().__init__()
        self.last_gaze_data = GazeData()
        self.angular_threshold_degree = angular_speed_threshold_degree
        self.invalid_gaze_data_count = 0

    def process_sample(self, gaze_data: GazeData):
        if self.last_gaze_data.combined_eye_gaze_data.gaze_point_valid and gaze_data.combined_eye_gaze_data.gaze_point_valid: # if both gaze data valid

            speed = angular_velocity_between_vectors_degrees(
                self.last_gaze_data.combined_eye_gaze_data.gaze_direction,
                gaze_data.combined_eye_gaze_data.gaze_direction,
                time_delta=gaze_data.timestamp - self.last_gaze_data.timestamp)
            # print(speed)
            if speed <= self.angular_threshold_degree:
                gaze_data.gaze_type = GazeType.FIXATION
            else:
                gaze_data.gaze_type = GazeType.SACCADE
        else:
            pass
            # self.invalid_gaze_data_count += 1
            # print('invalid gaze data:', self.invalid_gaze_data_count)
        self.last_gaze_data = gaze_data

        return gaze_data

    def reset_data_processor(self):
        self.last_gaze_data = GazeData()


'''
 The 0,0 coordinate of the image is the top left corner of the image
 The 0,0 coordinate of the display area is the top left corner of the display area
'''


def tobii_gaze_on_display_area_to_image_matrix_index(
        image_center_x,
        image_center_y,

        image_width,
        image_height,

        screen_width,
        screen_height,

        gaze_on_display_area_x,
        gaze_on_display_area_y):

    image_top_left_x_coordinate = image_center_x - image_width / 2
    image_top_left_y_coordinate = image_center_y + image_height / 2

    gaze_on_display_area_x_coordinate = screen_width * (gaze_on_display_area_x-0.5)
    gaze_on_display_area_y_coordinate = screen_height * (0.5-gaze_on_display_area_y)

    gaze_on_image_x_index = gaze_on_display_area_x_coordinate - image_top_left_x_coordinate
    gaze_on_image_y_index = image_top_left_y_coordinate - gaze_on_display_area_y_coordinate

    coordinate = np.array([gaze_on_image_y_index, gaze_on_image_x_index], dtype=np.int_) # the matrix location of the gaze point

    return coordinate # the matrix location of the gaze point

def gaze_point_on_image_valid(matrix_shape, coordinate):
    if coordinate[0] < 0 or coordinate[0] > matrix_shape[0]-1:
        return False
    if coordinate[1] < 0 or coordinate[1] > matrix_shape[1]-1:
        return False
    return True


