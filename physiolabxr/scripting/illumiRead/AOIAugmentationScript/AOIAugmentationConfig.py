from enum import Enum
import numpy as np
import os
import pickle





ReportCleanedImageInfoFilePath = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\experiment_data\report_cleaned_image_info.pkl'
#ReportCleanedImageInfoFilePath = r'D:\HaowenWei\Rena\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\experiment_data\report_cleaned_image_info.pkl'
SubImgaeHandlerFilePath = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\illumiRead\AOIAugmentationScript\data\subimage_handler.pkl'

screen_width = 1920
screen_height = 1080

image_on_screen_max_width = 1400
image_on_screen_max_height = 750

image_center_x = 0
image_center_y = 0

#########################################################################################

PracticeBlockImages = ["9071_OD_2021_widefield_report"]

TestBlockImages = [
    "RLS_097_OD_TC",
    "RLS_006_OD_TC",
    "RLS_043_OD_TC",
    # "RLS_083_OD_TC",
    # "8918_OS_2021_widefield_report",
    # "RLS_073_OD_TC",
    # "RLS_033_OS_TC",
    # "RLS_096_OS_TC",
    # "8981_OS_2021_widefield_report",
    # "RLS_073_OS_TC",
    # "RLS_086_OS_TC",
    # "RLS_060_OS_TC",
    # "RLS_085_OS_TC",
    # "RLS_079_OD_TC",
    # "RLS_082_OD_TC",
    # "9025_OD_2021_widefield_report",

    "RLS_083_OD_TC",
    "RLS_086_OS_TC",
    "RLS_045_OD_TC",
    # "RLS_006_OD_TC",
    # "RLS_092_OS_TC",
    # "8918_OS_2021_widefield_report",
    # "RLS_079_OD_TC",
    # "RLS_097_OD_TC",
    # "RLS_085_OS_TC",
    # "RLS_082_OD_TC",
    # "RLS_023_OD_TC",
    # "RLS_092_OS_TC",
    # "RLS_036_OS_TC",
    # "RLS_006_OD_TC",
    # "RLS_073_OD_TC",
    # "9025_OD_2021_widefield_report"
]



class AOIAugmentationScriptParams:
    AOIAugmentationInteractiveStateSubImagePlotWhenUpdate = "AOIAugmentationInteractiveStateSubImagePlotWhenUpdate"
    AOIAugmentationInteractiveStateNormalizeSubImage = "AOIAugmentationInteractiveStateNormalizeSubImage"




class EventMarkerLSLStreamInfo:
    StreamName = "AOIAugmentationEventMarkerLSL"
    StreamType = "EventMarker"
    StreamID = "1"
    ChannelNum = 6
    NominalSamplingRate = 1

    BlockChannelIndex = 0
    ExperimentStateChannelIndex = 1
    ImageIndexChannelIndex = 2
    AOIAugmentationInteractionStartEndMarker = 3
    ToggleVisualCueVisibilityMarker = 4
    UpdateVisualCueMarker = 5
    VisualCueHistorySelectedMarker = 6


class GazeDataLSLStreamInfo:
    StreamName = "TobiiProFusionUnityLSL"
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


class AOIAugmentationAttentionHeatmapLSLStreamInfo:
    StreamName = "AOIAugmentationAttentionHeatmapLSLStreamStream"
    StreamType = "AOIHeatmap"
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

    ResNetAOIAugmentationInstructionState = 12
    ResNetAOIAugmentationState = 13

    FeedbackState = 14  # not used anymore
    EndState = 15


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
