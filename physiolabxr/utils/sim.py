import random
import time
import numpy as np

from physiolabxr.configs import config, config_signal


def sim_openBCI_eeg():
    return np.random.uniform(low=0.0, high=1.0, size=(config.OPENBCI_EEG_CHANNEL_SIZE, 1))

def sim_unityLSL():
    return np.random.uniform(low=0.0, high=1.0, size=(config.UNITY_LSL_CHANNEL_SIZE, 1))

def sim_inference():
    return [[-1. for _ in range(config.INFERENCE_CLASS_NUM)]]

def simulateData():
    detObj = dict()

    num_points = random.randint(1, 40)
    detObj['x'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['y'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['z'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))
    detObj['doppler'] = np.random.uniform(low=0.0, high=1.0, size=(num_points,))

    time.sleep(0.033)

    return True, 0, detObj


def sim_heatmap(shape: list):
    return np.random.random(shape).astype(np.float16)


def sim_detected_points():
    num_points = random.randint(1, 40)
    return np.transpose(np.asarray([np.random.uniform(low=0.0, high=1.0, size=(num_points,)),
                                    np.random.uniform(low=0.0, high=1.0, size=(num_points,)),
                                    np.random.uniform(low=0.0, high=1.0, size=(num_points,)),
                                    np.random.uniform(low=0.0, high=1.0, size=(num_points,))]))


def sim_imp():
    return np.random.uniform(low=0.0, high=1.0, size=(config_signal.range_bins,))


def sim_uwb():
    return np.reshape(np.random.uniform(low=0.0, high=10000, size=(65 * 2)), newshape=(65, 2))


def sim_leap():
    return '1.0 2.0 3.0 4.0 5.0', sim_heatmap((256, 256))


def sim_xe4thru(length=100):
    return np.random.uniform(low=0.0, high=1.0, size=(length,))
