from enum import Enum
import numpy as np
import os
import pickle





ReportCleanedImageInfoFilePath = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\experiment_data\report_cleaned_image_info.pkl'
#ReportCleanedImageInfoFilePath = r'D:\HaowenWei\Rena\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\experiment_data\report_cleaned_image_info.pkl'

screen_width = 1920
screen_height = 1080

image_on_screen_width = 1800
image_on_screen_height = 900

image_center_x = 0
image_center_y = 0

#########################################################################################

PracticeBlockImages = ["9175_OS_2021_widefield_report.png",
                       "9172_OD_2021_widefield_report.png",
                       "RLS_023_OS_TC.jpg"]

TestBlockImages = ["9061_OS_2021_widefield_report.png",
                   "RLS_064_OS_TC.jpg",
                   "RLS_078_OS_TC.jpg"]




class EventMarkerLSLStreamInfo:
    StreamName = "AOIAugmentationEventMarkerLSLOutlet"
    StreamType = "EventMarker"
    StreamID = "1"
    ChannelNum = 4
    NominalSamplingRate = 1
    BlockChannelIndex = 0
    ExperimentStateChannelIndex = 1
    ImageIndexChannelIndex = 2
    UserInputsChannelIndex = 3


class GazeDataLSLStreamInfo:
    StreamName = "TobiiProFusionUnityLSLOutlet"
    StreamType = "GazeData"
    StreamID = "2"
    ChannelNum = 51
    NominalSamplingRate = 250


# class NoAOIAugmentationStateLSLStreamInfo:
#     # we do not need to send any data for this state
#     pass


# class StaticAOIAugmentationStateLSLStreamInfo:
#     StreamName = "StaticAOIAugmentationStateLSLInlet"
#     StreamType = "AttentionData"
#     StreamID = "3"
#     ChannelNum = int(attention_grid_shape[0] * attention_grid_shape[1])
#     NominalSamplingRate = 250
#
#
# class InteractiveAOIAugmentationStateLSLStreamInfo:
#     StreamName = "InteractiveAOIAugmentationStateLSLInlet"
#     StreamType = "AttentionData"
#     StreamID = "4"
#     ChannelNum = int(attention_grid_shape[0] * attention_grid_shape[1])
#     NominalSamplingRate = 250
#
#
class AOIAugmentationGazeAttentionMapLSLStreamInfo:
    StreamName = "AOIAugmentationGazeAttentionMapLSLOutlet"
    StreamType = "AttentionData"
    StreamID = "5"
    ChannelNum = int(32*32)
    NominalSamplingRate = 250


# class EventMarkerLSLInletInfo(Enum):
#     StreamName = "AOIAugmentationEventMarkerLSLInlet"
#     fs = None
#     channel_count = 2
#     channel_format = "float32"
#
#
# class TobiiProFusionUnityLSLOutlet(Enum):


class AOIAugmentationAttentionContourLSLStreamInfo:
    StreamName = "AOIAugmentationAttentionContourStream"
    StreamType = "AOIContour"
    StreamID = "3"
    ChannelNum = 1024
    NominalSamplingRate = 1


class ExperimentState(Enum):
    CalibrationState = 1
    StartState = 2
    IntroductionInstructionState = 3
    PracticeInstructionState = 4
    TestInstructionState = 5
    NoAOIAugmentationInstructionState = 6
    NoAOIAugmentationState = 7
    StaticAOIAugmentationInstructionState = 8
    StaticAOIAugmentationState = 9
    InteractiveAOIAugmentationInstructionState = 10
    InteractiveAOIAugmentationState = 11
    FeedbackState = 12
    EndState = 13


class ExperimentBlock(Enum):
    InitBlock = 0
    StartBlock = 1
    IntroductionBlock = 2
    PracticeBlock = 3
    TestBlock = 4
    EndBlock = 5

class UserInputTypes(Enum):

        AOIAugmentationInteractionStateUpdateCueKeyPressed = 1



# class NetworkConfig(Enum):
#     ZMQPortNumber = 6667


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
