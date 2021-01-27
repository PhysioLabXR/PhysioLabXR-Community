import random
import time
import numpy as np
import config

def sim_openBCI_eeg():
    return np.random.uniform(low=0.0, high=1.0, size=(1, config.OPENBCI_EEG_CHANNEL_SIZE))

def sim_unityLSL():
    return np.random.uniform(low=0.0, high=1.0, size=(1, config.UNITY_LSL_CHANNEL_SIZE))
