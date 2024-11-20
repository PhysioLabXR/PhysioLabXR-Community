from collections import deque

import numpy as np

import os
import os.path
import pickle
import sys
import time
from datetime import datetime


from itertools import groupby

from numba.core.cgutils import printf

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


if __name__=="__main__":

    # zmq gaze capture
    subtopic = "GazeCapture"
    sub_tcpAddress = "tcp://localhost:5556"
    context = zmq.Context()
    gaze_capture_sub_socket = context.socket(zmq.SUB)
    gaze_capture_sub_socket.connect(sub_tcpAddress)
    gaze_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, subtopic)

    # sockets connected and start the streaming
    printf(f'Sockets connected, entering image loop. ')
    while True:
        try:
            received = gaze_capture_sub_socket.recv_multipart()


        #     predtion
        #     publisher:

        except KeyboardInterrupt:
            print("KeyboardInterrupt")
