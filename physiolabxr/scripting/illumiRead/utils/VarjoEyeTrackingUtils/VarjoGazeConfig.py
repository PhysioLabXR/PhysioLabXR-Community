from enum import Enum
import numpy as np


class IndexClass(int, Enum):
    pass


class VarjoLSLChannelInfo(IndexClass):

    # Gaze data frame number
    FrameNumber = 0

    # Gaze data capture time (nanoseconds)
    CaptureTime = 1

    # Log time (milliseconds)
    LogTime = 2

    # HMD
    HMDLocalPositionX = 3
    HMDLocalPositionY = 4
    HMDLocalPositionZ = 5

    HMDLocalRotationX = 6
    HMDLocalRotationY = 7
    HMDLocalRotationZ = 8

    # Combined gaze
    CombinedGazeValid = 9

    CombinedGazeForwardX = 10
    CombinedGazeForwardY = 11
    CombinedGazeForwardZ = 12

    CombinedGazeOriginX = 13
    CombinedGazeOriginY = 14
    CombinedGazeOriginZ = 15

    # IDP
    InterPupillaryDistanceInMM = 16

    # Left eye
    LeftEyeGazeValid = 17

    LeftGazeForwardX = 18
    LeftGazeForwardY = 19
    LeftGazeForwardZ = 20

    LeftGazeOriginX = 21
    LeftGazeOriginY = 22
    LeftGazeOriginZ = 23

    LeftPupilIrisDiameterRatio = 24
    LeftPupilDiameterInMM = 25
    LeftIrisDiameterInMM = 26

    # Right eye
    RightEyeGazeValid = 27

    RightGazeForwardX = 28
    RightGazeForwardY = 29
    RightGazeForwardZ = 30

    RightGazeOriginX = 31
    RightGazeOriginY = 32
    RightGazeOriginZ = 33

    RightPupilIrisDiameterRatio = 34
    RightPupilDiameterInMM = 35
    RightIrisDiameterInMM = 36

    # focus
    FocusDistance = 37
    FocusStability = 38

    def __int__(self) -> int:
        return int.__int__(self)
