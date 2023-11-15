from enum import Enum
import numpy as np


class IndexClass(int, Enum):
    pass


class VarjoLSLChannelInfo(IndexClass):
    # Gaze data frame number
    GazeDataFrameNumber = 0

    # Gaze data capture time (nanoseconds)
    GazeCaptureTime = 1

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

    # CombinedGazeRayScreenDirectionX = 0
    # CombinedGazeRayScreenDirectionY = 1
    # CombinedGazeRayScreenDirectionZ = 2
    # CombinedGazeRayScreenOriginX = 3
    # CombinedGazeRayScreenOriginY = 4
    # CombinedGazeRayScreenOriginZ = 5
    # CombinedGazeRayScreenOriginValid = 6
    # LeftGazeOriginInTrackBoxCoordinatesX = 7
    # LeftGazeOriginInTrackBoxCoordinatesY = 8
    # LeftGazeOriginInTrackBoxCoordinatesZ = 9
    # LeftGazeOriginInUserCoordinatesX = 10
    # LeftGazeOriginInUserCoordinatesY = 11
    # LeftGazeOriginInUserCoordinatesZ = 12
    # LeftGazeOriginValid = 13
    # LeftGazePointInUserCoordinatesX = 14
    # LeftGazePointInUserCoordinatesY = 15
    # LeftGazePointInUserCoordinatesZ = 16
    # LeftGazePointOnDisplayAreaX = 17
    # LeftGazePointOnDisplayAreaY = 18
    # LeftGazePointValid = 19
    # LeftGazeRayScreenDirectionX = 20
    # LeftGazeRayScreenDirectionY = 21
    # LeftGazeRayScreenDirectionZ = 22
    # LeftGazeRayScreenOriginX = 23
    # LeftGazeRayScreenOriginY = 24
    # LeftGazeRayScreenOriginZ = 25
    # LeftPupilDiameter = 26
    # LeftPupilDiameterValid = 27
    # RightGazeOriginInTrackBoxCoordinatesX = 28
    # RightGazeOriginInTrackBoxCoordinatesY = 29
    # RightGazeOriginInTrackBoxCoordinatesZ = 30
    # RightGazeOriginInUserCoordinatesX = 31
    # RightGazeOriginInUserCoordinatesY = 32
    # RightGazeOriginInUserCoordinatesZ = 33
    # RightGazeOriginValid = 34
    # RightGazePointInUserCoordinatesX = 35
    # RightGazePointInUserCoordinatesY = 36
    # RightGazePointInUserCoordinatesZ = 37
    # RightGazePointOnDisplayAreaX = 38
    # RightGazePointOnDisplayAreaY = 39
    # RightGazePointValid = 40
    # RightGazeRayScreenDirectionX = 41
    # RightGazeRayScreenDirectionY = 42
    # RightGazeRayScreenDirectionZ = 43
    # RightGazeRayScreenOriginX = 44
    # RightGazeRayScreenOriginY = 45
    # RightGazeRayScreenOriginZ = 46
    # RightPupilDiameter = 47
    # RightPupilDiameterValid = 48
    # OriginalGazeDeviceTimeStamp = 49
    # OriginalGazeSystemTimeStamp = 50

    def __int__(self) -> int:
        return int.__int__(self)
