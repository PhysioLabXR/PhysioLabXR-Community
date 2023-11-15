from enum import Enum
import numpy as np
import os
import pickle


class EventMarkerLSLStreamInfo:
    StreamName = "illumiReadSwypeEventMarkerLSL"
    StreamType = "EventMarker"
    StreamID = "1"
    ChannelNum = 3
    NominalSamplingRate = 1
    BlockChannelIndex = 0
    ExperimentStateChannelIndex = 1
    UserInputsChannelIndex = 2


class GazeDataLSLStreamInfo:
    StreamName = "VarjoEyeTrackingLSL"
    StreamType = "GazeData"
    StreamID = "2"
    ChannelNum = 35
    NominalSamplingRate = 200


class ExperimentState(Enum):
    InitState = 0

    CalibrationState = 1
    StartState = 2
    IntroductionInstructionState = 3

    KeyboardDewellTimeIntroductionState = 4
    KeyboardDewellTimeState = 5

    KeyboardClickIntroductionState = 6
    KeyboardClickState = 7

    KeyboardIllumiReadSwypeIntroductionState = 8
    KeyboardIllumiReadSwypeState = 9

    KeyboardFreeSwitchInstructionState = 10
    KeyboardFreeSwitchState = 11

    FeedbackState = 12

    EndState = 13


class ExperimentBlock(Enum):
    InitBlock = 0
    StartBlock = 1
    IntroductionBlock = 2
    PracticeBlock = 3
    TrainBlock = 4
    TestBlock = 5
    EndBlock = 6

class UserInputTypes(Enum):
    ButtonPress = 1

from enum import Enum


class IndexClass(int, Enum):
    pass


class VarjoEyeTrackingChannel(IndexClass):
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
