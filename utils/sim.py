import random
import time
import numpy as np
import config

def sim_openBCI_eeg():
    return np.random.uniform(low=0.0, high=1.0, size=(config.OPENBCI_EEG_CHANNEL_SIZE, 1))

def sim_unityLSL():
    return np.random.uniform(low=0.0, high=1.0, size=(config.UNITY_LSL_CHANNEL_SIZE, 1))

def sim_inference():
    return [[-1. for _ in range(config.INFERENCE_CLASS_NUM)]]