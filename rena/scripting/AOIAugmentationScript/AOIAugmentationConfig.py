from enum import Enum
import numpy as np

screen_width = 1920
screen_height = 1080

patch_grid_width = 50
patch_grid_height = 25

image_width = 2000
image_height = 1000

image_center_x = 0
image_center_y = 0

image_shape = np.array([500, 1000])
attention_patch_shape = np.array([20, 20])
attention_grid_shape = np.array([25, 50])
image_on_screen_shape = np.array([1200, 2400])


class EventMarkerLSLOutlet:
    StreamName = "AOIAugmentationEventMarkerLSLOutlet"
    StreamType = "EventMarker"
    StreamID = "1"
    ChannelNum = 3
    NominalSamplingRate = 1
    BlockChannelIndex = 0
    ExperimentStateChannelIndex = 1
    ReportLabelChannelIndex = 2



class GazeDataLSLOutlet:
    StreamName = "TobiiProFusionUnityLSLOutlet"
    StreamType = "GazeData"
    StreamID = "2"
    ChannelNum = 51
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
