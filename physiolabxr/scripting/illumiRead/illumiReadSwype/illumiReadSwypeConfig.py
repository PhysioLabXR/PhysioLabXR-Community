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
    ChannelNum = 39
    NominalSamplingRate = 200


class UserInputLSLStreamInfo:
    StreamName = "illumiReadSwypeUserInputLSL"
    StreamType = "UserInput"
    StreamID = "3"
    ChannelNum = 11
    NominalSamplingRate = 80

    GazeHitKeyboardBackgroundChannelIndex = 0
    KeyboardBackgroundHitPointLocalXChannelIndex = 1
    KeyboardBackgroundHitPointLocalYChannelIndex = 2
    KeyboardBackgroundHitPointLocalZChannelIndex = 3
    GazeHitKeyChannelIndex = 4
    KeyHitPointLocalXChannelIndex = 5
    KeyHitPointLocalYChannelIndex = 6
    KeyHitPointLocalZChannelIndex = 7
    KeyHitIndexChannelIndex = 8
    UserInputButton1ChannelIndex = 9
    UserInputButton2ChannelIndex = 10
    UserInputButton3ChannelIndex = 11


class illumiReadSwypeKeyboardSuggestionStripLSLStreamInfo:
    StreamName = "illumiReadSwypeKeyboardSuggestionStripLSL"
    StreamType = "KeyboardSuggestionStrip"
    StreamID = "4"
    ChannelNum = 1024
    NominalSamplingRate = 1
    
# class KeyboardContextLSLStreamInfo:
#     StreamName = "illumiReadSwypeKeyboardContextLSL"
#     StreamType = "KeyboardContext"
#     StreamID = "5"
#     ChannelNum = 1
#     NominalSamplingRate = 80
    
#     KeyboardContextChannelIndex = 0


KeyIDIndexDict = {
    0: None,
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
    6: "F",
    7: "G",
    8: "H",
    9: "I",
    10: "J",
    11: "K",
    12: "L",
    13: "M",
    14: "N",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",

    27: "'",
    28: "."

}

KeyIndexIDDict = {
    None: 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,

    "'": 27,
    ".": 28
}


class ExperimentState(Enum):
    # InitState = 0
    #
    # CalibrationState = 1
    # StartState = 2
    # IntroductionInstructionState = 3
    # IntroductionEegState =4
    #
    # KeyboardDewellTimeIntroductionState = 5
    # KeyboardDewellTimeState = 6
    #
    # KeyboardClickIntroductionState = 7
    # KeyboardClickState = 8
    #
    # KeyboardIllumiReadSwypeIntroductionState = 9
    # KeyboardIllumiReadSwypeState = 10
    #
    # KeyboardFreeSwitchInstructionState = 11
    # KeyboardFreeSwitchState = 12
    #
    # FeedbackState = 13
    #
    # EndState = 14
    # EegState = 15


    InitState = 0

    CalibrationState = 1
    StartState = 2
    IntroductionInstructionState = 3
    IntroductionEegState = 4
    EegState = 5

    KeyboardDewellTimeIntroductionState = 6
    KeyboardDewellTimeState = 7

    KeyboardClickIntroductionState = 8
    KeyboardClickState = 9

    KeyboardIllumiReadSwypeIntroductionState = 10
    KeyboardIllumiReadSwypeState = 11

    KeyboardFreeSwitchInstructionState = 12
    KeyboardFreeSwitchState = 13

    KeyboardIllumiReadSwypeIntroductionState_noeeg = 14
    KeyboardIllumiReadSwypeState_noeeg = 15

    FeedbackState = 16
    EndState = 17

    # FeedbackState = 14
    # EndState = 15



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
