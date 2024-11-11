from collections import deque

import numpy as np

import os
import os.path
import pickle
import sys
import time
from datetime import datetime


from itertools import groupby
from physiolabxr.scripting.RenaScript import RenaScript

# necessary packages for physiolabXR
from physiolabxr.scripting.GazeHotkey.GazeHotkeyProject import GazeHotkeyConfig
from physiolabxr.utils.buffers import DataBuffer

# import the necesary packages used for machine learning
import pandas as pd
from nltk import RegexpTokenizer
from physiolabxr.rpc.decorator import rpc,async_rpc
import cv2
import zmq
import numpy as np
import struct

# #class of renascript
# class GazeHotkeyProject(RenaScript):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         # setup the init file for python backend
#
#     def init(self):
#         pass
#
#     def loop(self):
#         pass


if __name__=="__main__":
    pass