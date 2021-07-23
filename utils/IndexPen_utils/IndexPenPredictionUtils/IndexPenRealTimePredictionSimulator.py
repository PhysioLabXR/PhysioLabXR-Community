import time
from collections import deque

import numpy as np
import tensorflow as tf

from utils.data_utils import is_broken_frame, clutter_removal

class IndexPenRealTimePredictionSimulator:
    def __init__(self, model_path=None, data_path=None):
        self.data_path = data_path