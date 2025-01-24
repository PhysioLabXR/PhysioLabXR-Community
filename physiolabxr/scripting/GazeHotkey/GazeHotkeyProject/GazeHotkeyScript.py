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

# ZMQ camera socket
def get_cam_socket(sub_tcpAddress, topic: str):
    context = zmq.Context()
    cam_capture_sub_socket = context.socket(zmq.SUB)
    cam_capture_sub_socket.connect(sub_tcpAddress)
    cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    return cam_capture_sub_socket

# ZMQ decode received gaze trace
def receive_decode_info(socket):
    received = socket.recv_multipart()

    player_position = received[1]
    player_rotation = received[2]
    UIHitLocal = received[3]
    KeyHitLocal = received[4]
    currentTextUI = received[5]

    print(f"Player Position: {player_position}, Player Rotation: {player_rotation}, UI Hit Local: {UIHitLocal}, Key Hit Local: {KeyHitLocal}, Current Text UI: {currentTextUI}")
    # decode the received data

    return

if __name__=="__main__":
    camera_cam_socket = get_cam_socket("tcp://localhost:5556", 'CamGazeHitPosition')

    print("Sockets connected, entering streaming loop.")
    while True:
        try:
            receive_decode_info(camera_cam_socket)
        except KeyboardInterrupt:
            print("Exiting the streaming loop.")
            break


