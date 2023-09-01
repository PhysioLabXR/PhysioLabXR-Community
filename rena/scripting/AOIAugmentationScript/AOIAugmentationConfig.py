from enum import Enum
import numpy as np

screen_width = 1920
screen_height = 1080

# patch_grid_width = 50
# patch_grid_height = 25

image_on_screen_width = 1900
image_on_screen_height = 950


image_center_x = 0
image_center_y = 0


image_shape = np.array([500, 1000], dtype=np.int32)
attention_patch_shape = np.array([20, 20], dtype=np.int32)
attention_grid_shape = np.array([image_shape[0]//attention_patch_shape[0], image_shape[1]//attention_patch_shape[1]], dtype=np.int32)

# attention_grid_shape = np.array([25, 50], dtype=np.int32)
image_on_screen_shape = np.array([image_on_screen_height, image_on_screen_width], dtype=np.int32)
image_scaling_factor = np.array([image_on_screen_shape[0]/image_shape[0], image_on_screen_shape[1]/image_shape[1]], dtype=np.float32)


class EventMarkerLSLStreamInfo:
    StreamName = "AOIAugmentationEventMarkerLSLOutlet"
    StreamType = "EventMarker"
    StreamID = "1"
    ChannelNum = 3
    NominalSamplingRate = 1
    BlockChannelIndex = 0
    ExperimentStateChannelIndex = 1
    ReportLabelChannelIndex = 2


class GazeDataLSLStreamInfo:
    StreamName = "TobiiProFusionUnityLSLOutlet"
    StreamType = "GazeData"
    StreamID = "2"
    ChannelNum = 51
    NominalSamplingRate = 250

class StaticAOIAugmentationStateLSLStreamInfo:
    StreamName = "StaticAOIAugmentationStateLSLInlet"
    StreamType = "AttentionData"
    StreamID = "3"
    ChannelNum = int(attention_grid_shape[0]*attention_grid_shape[1])
    NominalSamplingRate = 250

class AOIAugmentationGazeAttentionMapLSLStreamInfo:
    StreamName = "AOIAugmentationGazeAttentionMapLSLOutlet"
    StreamType = "AttentionData"
    StreamID = "4"
    ChannelNum = int(attention_grid_shape[0]*attention_grid_shape[1])
    NominalSamplingRate = 250




# class EventMarkerLSLInletInfo(Enum):
#     StreamName = "AOIAugmentationEventMarkerLSLInlet"
#     fs = None
#     channel_count = 2
#     channel_format = "float32"
#
#
# class TobiiProFusionUnityLSLOutlet(Enum):


class ExperimentState(Enum):
    CalibrationState = 1
    StartState = 2
    IntroductionInstructionState = 3
    PracticeInstructionState = 4
    NoAOIAugmentationInstructionState = 5
    NoAOIAugmentationState = 6
    StaticAOIAugmentationInstructionState = 7
    StaticAOIAugmentationState = 8
    InteractiveAOIAugmentationInstructionState = 9
    InteractiveAOIAugmentationState = 10
    FeedbackState = 11
    EndState = 12


class ExperimentBlock(Enum):
    InitBlock = 0
    StartBlock = 1
    IntroductionBlock = 2
    PracticeBlock = 3
    ExperimentBlock = 4
    EndBlock = 5


class NetworkConfig(Enum):
    ZMQPortNumber = 6667


# channel_info = [
#     "CombinedGazeRayScreenDirectionX",
#     "CombinedGazeRayScreenDirectionY",
#     "CombinedGazeRayScreenDirectionZ",
#     "CombinedGazeRayScreenOriginX",
#     "CombinedGazeRayScreenOriginY",
#     "CombinedGazeRayScreenOriginZ",
#     "CombinedGazeRayScreenOriginValid",
#     "LeftGazeOriginInTrackBoxCoordinatesX",
#     "LeftGazeOriginInTrackBoxCoordinatesY",
#     "LeftGazeOriginInTrackBoxCoordinatesZ",
#     "LeftGazeOriginInUserCoordinatesX",
#     "LeftGazeOriginInUserCoordinatesY",
#     "LeftGazeOriginInUserCoordinatesZ",
#     "LeftGazeOriginValid",
#     "LeftGazePointInUserCoordinatesX",
#     "LeftGazePointInUserCoordinatesY",
#     "LeftGazePointInUserCoordinatesZ",
#     "LeftGazePointOnDisplayAreaX",
#     "LeftGazePointOnDisplayAreaY",
#     "LeftGazePointValid",
#     "LeftGazeRayScreenDirectionX",
#     "LeftGazeRayScreenDirectionY",
#     "LeftGazeRayScreenDirectionZ",
#     "LeftGazeRayScreenOriginX",
#     "LeftGazeRayScreenOriginY",
#     "LeftGazeRayScreenOriginZ",
#     "LeftPupileDiameter",
#     "LeftPupileDiameterValid",
#     "RightGazeOriginInTrackBoxCoordinatesX",
#     "RightGazeOriginInTrackBoxCoordinatesY",
#     "RightGazeOriginInTrackBoxCoordinatesZ",
#     "RightGazeOriginInUserCoordinatesX",
#     "RightGazeOriginInUserCoordinatesY",
#     "RightGazeOriginInUserCoordinatesZ",
#     "RightGazeOriginValid",
#     "RightGazePointInUserCoordinatesX",
#     "RightGazePointInUserCoordinatesY",
#     "RightGazePointInUserCoordinatesZ",
#     "RightGazePointOnDisplayAreaX",
#     "RightGazePointOnDisplayAreaY",
#     "RightGazePointValid",
#     "RightGazeRayScreenDirectionX",
#     "RightGazeRayScreenDirectionY",
#     "RightGazeRayScreenDirectionZ",
#     "RightGazeRayScreenOriginX",
#     "RightGazeRayScreenOriginY",
#     "RightGazeRayScreenOriginZ",
#     "RightPupilDiameter",
#     "RightPupilDiameterValid",
#     "OriginalGazeDeviceTimeStamp",
#     "OriginalGazeSystemTimeStamp"
# ]
# import enum

from enum import Enum


class IndexClass(int, Enum):
    pass


class TobiiProFusionChannel(IndexClass):
    CombinedGazeRayScreenDirectionX = 0
    CombinedGazeRayScreenDirectionY = 1
    CombinedGazeRayScreenDirectionZ = 2
    CombinedGazeRayScreenOriginX = 3
    CombinedGazeRayScreenOriginY = 4
    CombinedGazeRayScreenOriginZ = 5
    CombinedGazeRayScreenOriginValid = 6
    LeftGazeOriginInTrackBoxCoordinatesX = 7
    LeftGazeOriginInTrackBoxCoordinatesY = 8
    LeftGazeOriginInTrackBoxCoordinatesZ = 9
    LeftGazeOriginInUserCoordinatesX = 10
    LeftGazeOriginInUserCoordinatesY = 11
    LeftGazeOriginInUserCoordinatesZ = 12
    LeftGazeOriginValid = 13
    LeftGazePointInUserCoordinatesX = 14
    LeftGazePointInUserCoordinatesY = 15
    LeftGazePointInUserCoordinatesZ = 16
    LeftGazePointOnDisplayAreaX = 17
    LeftGazePointOnDisplayAreaY = 18
    LeftGazePointValid = 19
    LeftGazeRayScreenDirectionX = 20
    LeftGazeRayScreenDirectionY = 21
    LeftGazeRayScreenDirectionZ = 22
    LeftGazeRayScreenOriginX = 23
    LeftGazeRayScreenOriginY = 24
    LeftGazeRayScreenOriginZ = 25
    LeftPupilDiameter = 26
    LeftPupilDiameterValid = 27
    RightGazeOriginInTrackBoxCoordinatesX = 28
    RightGazeOriginInTrackBoxCoordinatesY = 29
    RightGazeOriginInTrackBoxCoordinatesZ = 30
    RightGazeOriginInUserCoordinatesX = 31
    RightGazeOriginInUserCoordinatesY = 32
    RightGazeOriginInUserCoordinatesZ = 33
    RightGazeOriginValid = 34
    RightGazePointInUserCoordinatesX = 35
    RightGazePointInUserCoordinatesY = 36
    RightGazePointInUserCoordinatesZ = 37
    RightGazePointOnDisplayAreaX = 38
    RightGazePointOnDisplayAreaY = 39
    RightGazePointValid = 40
    RightGazeRayScreenDirectionX = 41
    RightGazeRayScreenDirectionY = 42
    RightGazeRayScreenDirectionZ = 43
    RightGazeRayScreenOriginX = 44
    RightGazeRayScreenOriginY = 45
    RightGazeRayScreenOriginZ = 46
    RightPupilDiameter = 47
    RightPupilDiameterValid = 48
    OriginalGazeDeviceTimeStamp = 49
    OriginalGazeSystemTimeStamp = 50

    def __int__(self) -> int:
        return int.__int__(self)
