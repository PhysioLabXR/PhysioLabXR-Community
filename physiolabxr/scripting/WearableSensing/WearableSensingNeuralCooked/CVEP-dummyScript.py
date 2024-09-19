#Using 2 RPCs (train, decode) that gets called
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.cross_decomposition import CCA
from collections import deque
from enum import Enum
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.utils.buffers import DataBuffer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn import metrics
from physiolabxr.rpc.decorator import rpc, async_rpc

class NeuroCooked(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edi this function
        """
        super().__init__(*args, **kwargs)
    def init(self):