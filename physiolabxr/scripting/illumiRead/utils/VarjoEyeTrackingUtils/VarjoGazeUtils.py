import enum

import numpy as np

from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeConfig import VarjoLSLChannelInfo
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeData, GazeType


class VarjoGazeData(GazeData):
    def __init__(self,
                 frame_number=0,
                 capture_time=0,
                 log_time=0,

                 hmd_local_position=np.zeros(3),
                 hmd_local_rotation=np.zeros(3),

                 combined_gaze_valid=False,
                 combined_gaze_forward=np.zeros(3),
                 combined_gaze_origin=np.zeros(3),

                 inter_pupillary_distance_in_mm=0,

                 left_eye_gaze_valid=False,
                 left_gaze_forward=np.zeros(3),
                 left_gaze_origin=np.zeros(3),

                 left_pupil_iris_diameter_ratio=0,
                 left_pupil_diameter_in_mm=0,
                 left_iris_diameter_in_mm=0,

                 right_eye_gaze_valid=False,
                 right_gaze_forward=np.zeros(3),
                 right_gaze_origin=np.zeros(3),

                 right_pupil_iris_diameter_ratio=0,
                 right_pupil_diameter_in_mm=0,
                 right_iris_diameter_in_mm=0,

                 focus_distance=0,
                 focus_stability=0,

                 ):
        super().__init__()

        self.frame_number = frame_number
        self.capture_time = capture_time
        self.log_time = log_time

        self.hmd_local_position = hmd_local_position
        self.hmd_local_rotation = hmd_local_rotation

        self.combined_gaze_valid = combined_gaze_valid
        self.combined_gaze_forward = combined_gaze_forward
        self.combined_gaze_origin = combined_gaze_origin

        self.inter_pupillary_distance_in_mm = inter_pupillary_distance_in_mm

        self.left_eye_gaze_valid = left_eye_gaze_valid
        self.left_gaze_forward = left_gaze_forward
        self.left_gaze_origin = left_gaze_origin

        self.left_pupil_iris_diameter_ratio = left_pupil_iris_diameter_ratio
        self.left_pupil_diameter_in_mm = left_pupil_diameter_in_mm
        self.left_iris_diameter_in_mm = left_iris_diameter_in_mm

        self.right_eye_gaze_valid = right_eye_gaze_valid
        self.right_gaze_forward = right_gaze_forward
        self.right_gaze_origin = right_gaze_origin

        self.right_pupil_iris_diameter_ratio = right_pupil_iris_diameter_ratio
        self.right_pupil_diameter_in_mm = right_pupil_diameter_in_mm
        self.right_iris_diameter_in_mm = right_iris_diameter_in_mm

        self.focus_distance = focus_distance
        self.focus_stability = focus_stability

        # Added
        self.gaze_type = GazeType.UNDETERMINED
        self.timestamp = 0

    def construct_gaze_data_varjo(self, gaze_data_t, timestamp):
        self.frame_number = gaze_data_t[VarjoLSLChannelInfo.FrameNumber]

        self.capture_time = gaze_data_t[VarjoLSLChannelInfo.CaptureTime]

        self.log_time = gaze_data_t[VarjoLSLChannelInfo.LogTime]

        self.hmd_local_position = np.array([gaze_data_t[VarjoLSLChannelInfo.HMDLocalPositionX],
                                            gaze_data_t[VarjoLSLChannelInfo.HMDLocalPositionY],
                                            gaze_data_t[VarjoLSLChannelInfo.HMDLocalPositionZ]])

        self.hmd_local_rotation = np.array([gaze_data_t[VarjoLSLChannelInfo.HMDLocalRotationX],
                                            gaze_data_t[VarjoLSLChannelInfo.HMDLocalRotationY],
                                            gaze_data_t[VarjoLSLChannelInfo.HMDLocalRotationZ]])

        self.combined_gaze_valid = gaze_data_t[VarjoLSLChannelInfo.CombinedGazeValid]

        self.combined_gaze_forward = np.array([gaze_data_t[VarjoLSLChannelInfo.CombinedGazeForwardX],
                                               gaze_data_t[VarjoLSLChannelInfo.CombinedGazeForwardY],
                                               gaze_data_t[VarjoLSLChannelInfo.CombinedGazeForwardZ]])

        self.combined_gaze_origin = np.array([gaze_data_t[VarjoLSLChannelInfo.CombinedGazeOriginX],
                                              gaze_data_t[VarjoLSLChannelInfo.CombinedGazeOriginY],
                                              gaze_data_t[VarjoLSLChannelInfo.CombinedGazeOriginZ]])

        self.inter_pupillary_distance_in_mm = gaze_data_t[VarjoLSLChannelInfo.InterPupillaryDistanceInMM]

        self.left_eye_gaze_valid = gaze_data_t[VarjoLSLChannelInfo.LeftEyeGazeValid]

        self.left_gaze_forward = np.array([gaze_data_t[VarjoLSLChannelInfo.LeftGazeForwardX],
                                           gaze_data_t[VarjoLSLChannelInfo.LeftGazeForwardY],
                                           gaze_data_t[VarjoLSLChannelInfo.LeftGazeForwardZ]])

        self.left_gaze_origin = np.array([gaze_data_t[VarjoLSLChannelInfo.LeftGazeOriginX],
                                          gaze_data_t[VarjoLSLChannelInfo.LeftGazeOriginY],
                                          gaze_data_t[VarjoLSLChannelInfo.LeftGazeOriginZ]])

        self.left_pupil_iris_diameter_ratio = gaze_data_t[VarjoLSLChannelInfo.LeftPupilIrisDiameterRatio]
        self.left_pupil_diameter_in_mm = gaze_data_t[VarjoLSLChannelInfo.LeftPupilDiameterInMM]
        self.left_iris_diameter_in_mm = gaze_data_t[VarjoLSLChannelInfo.LeftIrisDiameterInMM]

        self.right_eye_gaze_valid = gaze_data_t[VarjoLSLChannelInfo.RightEyeGazeValid]

        self.right_gaze_forward = np.array([gaze_data_t[VarjoLSLChannelInfo.RightGazeForwardX],
                                            gaze_data_t[VarjoLSLChannelInfo.RightGazeForwardY],
                                            gaze_data_t[VarjoLSLChannelInfo.RightGazeForwardZ]])

        self.right_gaze_origin = np.array([gaze_data_t[VarjoLSLChannelInfo.RightGazeOriginX],
                                           gaze_data_t[VarjoLSLChannelInfo.RightGazeOriginY],
                                           gaze_data_t[VarjoLSLChannelInfo.RightGazeOriginZ]])

        self.right_pupil_iris_diameter_ratio = gaze_data_t[VarjoLSLChannelInfo.RightPupilIrisDiameterRatio]
        self.right_pupil_diameter_in_mm = gaze_data_t[VarjoLSLChannelInfo.RightPupilDiameterInMM]
        self.right_iris_diameter_in_mm = gaze_data_t[VarjoLSLChannelInfo.RightIrisDiameterInMM]

        self.focus_distance = gaze_data_t[VarjoLSLChannelInfo.FocusDistance]
        self.focus_stability = gaze_data_t[VarjoLSLChannelInfo.FocusStability]

        self.timestamp = timestamp

    def get_combined_eye_gaze_data_valid(self):
        return self.combined_gaze_valid

    def get_combined_eye_gaze_direction(self):
        return self.combined_gaze_forward

    def get_timestamp(self):
        return self.timestamp

    def set_gaze_type(self, gaze_type):
        self.gaze_type = gaze_type

    def get_gaze_type(self):
        return self.gaze_type
