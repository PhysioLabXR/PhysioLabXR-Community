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


class NetworkConfig(Enum):
    ZMQPortNumber = 6667

