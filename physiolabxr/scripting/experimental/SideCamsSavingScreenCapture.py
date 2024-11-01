"""
Realtime fixation detection based on patch similarity.

Also decode the camera frames, both color and depth and save them to
"""

import os.path
import time
from datetime import datetime

import cv2
import lpips
import pandas as pd
import zmq
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from physiolabxr.examples.Eyetracking.EyeUtils import prepare_image_for_sim_score, add_bounding_box
from physiolabxr.examples.Eyetracking.configs import *
import struct
import matplotlib.pyplot as plt

# fix detection parameters  #######################################
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
previous_img_patch = None
fixation_frame_counter = 0
distance = 0

# LSL detected fixations ########################################
outlet = StreamOutlet(StreamInfo("FixationDetection", 'FixationDetection', 3, 30, 'float32'))

# zmq camera capture fields #######################################

def get_cam_socket(sub_tcpAddress, topic: str):
    context = zmq.Context()
    cam_capture_sub_socket = context.socket(zmq.SUB)
    cam_capture_sub_socket.connect(sub_tcpAddress)
    cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    return cam_capture_sub_socket

def receive_decode_image(socket):
    received = socket.recv_multipart()
    timestamp = struct.unpack('d', received[1])[0]
    colorImagePNGBytes = received[2]
    depthImagePNGBytes = received[3]
    colorImg = np.frombuffer(colorImagePNGBytes, dtype='uint8').reshape((*image_shape[:2], 4))
    depthImg = np.frombuffer(depthImagePNGBytes, dtype='uint16').reshape((*image_shape[:2], 1))

    colorImg = colorImg[:, :, :3]
    colorImg = cv2.flip(colorImg, 0)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    depthImg = cv2.flip(depthImg, 0)

    return colorImg, depthImg, timestamp


right_cam_socket = get_cam_socket("tcp://localhost:5557", 'ColorDepthCamRight')
left_cam_socket = get_cam_socket("tcp://localhost:5558", 'ColorDepthCamLeft')
back_cam_socket = get_cam_socket("tcp://localhost:5559", 'ColorDepthCamBack')

# Disk Utilities Fields ########################################
capture_save_location = "C:/Users/LLINC-Lab/Documents/Recordings"


is_saving_captures = True
is_displaying = True

now = datetime.now()
dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
capture_save_location = os.path.join(capture_save_location, 'ReNaUnityCameraCapture_' + dt_string)
capture_right_path = os.path.join(capture_save_location, 'RightCamCapture')
capture_left_path = os.path.join(capture_save_location, 'LeftCamCapture')
capture_back_path = os.path.join(capture_save_location, 'BackCamCapture')

os.makedirs(capture_right_path, exist_ok=False)
os.makedirs(capture_left_path, exist_ok=False)
os.makedirs(capture_back_path, exist_ok=False)

frame_counter = 0


print('Sockets connected, entering image loop')
while True:
    try:
        fix_detection_sample = np.zeros(3) - 1

        right_cam_color, right_cam_depth, timestamp_right = receive_decode_image(right_cam_socket)
        left_cam_color, left_cam_depth, timestamp_left = receive_decode_image(left_cam_socket)
        back_cam_color, back_cam_depth, timestamp_back = receive_decode_image(back_cam_socket)

        # save the original image
        if is_saving_captures:
            cv2.imwrite(os.path.join(capture_right_path, f"{frame_counter}_t={struct.pack('f', timestamp_right)}.png"), right_cam_color)
            cv2.imwrite(os.path.join(capture_right_path, f"{frame_counter}_t={struct.pack('f', timestamp_right)}_depth.png"), right_cam_depth)

            cv2.imwrite(os.path.join(capture_left_path, f"{frame_counter}_t={struct.pack('f', timestamp_left)}.png"), left_cam_color)
            cv2.imwrite(os.path.join(capture_left_path, f"{frame_counter}_t={struct.pack('f', timestamp_left)}_depth.png"), left_cam_depth)

            cv2.imwrite(os.path.join(capture_back_path, f"{frame_counter}_t={struct.pack('f', timestamp_back)}.png"), back_cam_color)
            cv2.imwrite(os.path.join(capture_back_path, f"{frame_counter}_t={struct.pack('f', timestamp_back)}_depth.png"), back_cam_depth)

        frame_counter += 1

        if is_displaying:
            # concate the right color and depth side by side
            # left_cam_depth = cv2.normalize(left_cam_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # left_cam_depth = cv2.cvtColor(left_cam_depth, cv2.COLOR_GRAY2RGB)
            #
            # right_cam_depth = cv2.normalize(right_cam_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # right_cam_depth = cv2.cvtColor(right_cam_depth, cv2.COLOR_GRAY2RGB)
            #
            # back_cam_depth = cv2.normalize(back_cam_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # back_cam_depth = cv2.cvtColor(back_cam_depth, cv2.COLOR_GRAY2RGB)
            #
            # right = np.concatenate((right_cam_color, right_cam_depth), axis=1)
            # left = np.concatenate((left_cam_color, left_cam_depth), axis=1)
            # back = np.concatenate((back_cam_color, back_cam_depth), axis=1)

            # cv2.imshow('Left', left)
            # cv2.imshow('Back', back)

            cv2.imshow('Right', right_cam_color)
            cv2.imshow('RightDepth', right_cam_depth)

            cv2.imshow('Left', left_cam_color)
            cv2.imshow('LeftDepth', left_cam_depth)

            cv2.imshow('Back', back_cam_color)
            cv2.imshow('BackDepth', back_cam_depth)

            cv2.waitKey(delay=1)

    except KeyboardInterrupt:
        print('Stopped')

