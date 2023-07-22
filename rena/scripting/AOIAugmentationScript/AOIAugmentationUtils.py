import numpy as np

from enum import Enum


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


def generate_random_attention_matrix(grid_shape=(25, 50)):
    patch_num = grid_shape[0] * grid_shape[1]
    attention_matrix = np.random.random((patch_num, patch_num))
    return attention_matrix


if __name__ == '__main__':
    attention_matrix = generate_random_attention_matrix()
