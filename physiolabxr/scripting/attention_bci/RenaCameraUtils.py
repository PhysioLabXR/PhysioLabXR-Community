"""
Realtime fixation detection based on patch similarity.

Also decode the camera frames, both color and depth and save them to
"""
import csv
import json
import os.path
import struct
from datetime import datetime

import cv2
import numpy as np
import random
# import pandas as pd
import zmq

# from physiolabxr.examples.Eyetracking.EyeUtils import add_bounding_box, clip_bbox
# from physiolabxr.examples.Eyetracking.FixationDetection import fixation
from physiolabxr.examples.Eyetracking.configs import *
from physiolabxr.scripting.attention_bci.list_LVIS import COLORS,CLASSES

# zmq camera capture fields #######################################

def get_cam_socket(sub_tcpAddress, topic: str):
    context = zmq.Context()
    cam_capture_sub_socket = context.socket(zmq.SUB)
    cam_capture_sub_socket.connect(sub_tcpAddress)
    cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
    return cam_capture_sub_socket

def receive_decode_image(socket):

    # feature: use the zmq Noblock
    try:
        received = socket.recv_multipart(zmq.NOBLOCK)
    except zmq.Again:
        return None

    timestamp = struct.unpack('d', received[1])[0]
    colorImagePNGBytes = received[2]
    depthImagePNGBytes = received[3]
    item_bboxes = received[4]

    colorImg = np.frombuffer(colorImagePNGBytes, dtype='uint8').reshape((*image_shape[:2], 4))
    depthImg = np.frombuffer(depthImagePNGBytes, dtype='uint16').reshape((*image_shape[:2], 1))

    colorImg = colorImg[:, :, :3]
    colorImg = cv2.flip(colorImg, 0)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    depthImg = cv2.flip(depthImg, 0)

    item_bboxes = json.loads(item_bboxes)

    return colorImg, depthImg, timestamp, item_bboxes

def receive_fixation_decode(socket):
    # feature: use the zmq Noblock for fixation image receiving

    try:
        fixation_received = socket.recv_multipart(zmq.NOBLOCK)
    except zmq.Again:
        return None

    # TODO: the fixation camera is not on for now and the fixation_cam_socket.recev_multipart can not work properly
    timestamp = struct.unpack('d', fixation_received[1])[0]
    colorImagePNGBytes = fixation_received[2]
    depthImagePNGBytes = fixation_received[3]
    colorImg = np.frombuffer(colorImagePNGBytes, dtype='uint8').reshape((*image_shape[:2], 4))
    colorImg = colorImg[:, :, :3]
    colorImg = cv2.flip(colorImg, 0)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)

    depthImg = np.frombuffer(depthImagePNGBytes, dtype='uint16').reshape((*image_shape[:2], 1))
    depthImg = cv2.flip(depthImg, 0)

    gazed_item_index, gazed_item_dtn = struct.unpack('hh', fixation_received[5])
    gaze_info = fixation_received[6]
    gaze_x, gaze_y = struct.unpack('hh', gaze_info)  # the gaze coordinate
    gaze_y = image_shape[1] - gaze_y  # because CV's y zero is at the bottom of the screen

    # deprecated: should use the item bounding box given by the OVTR model
    item_bboxes = fixation_received[4]
    item_bboxes = json.loads(item_bboxes)

    return colorImg, depthImg, timestamp, item_bboxes, gaze_x, gaze_y



def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None, mask=None):
    # Plots one bounding box on image img
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3.5, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3.5, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
    return img


# feature: OVTR draw bbox function, only use this to test the bbox drawing, the labeling are problematic
def draw_bboxes(ori_img, bbox, identities=None, mask=None, offset=(0, 0)):
    img = ori_img
    for i, box in enumerate(bbox):
        if mask is not None and mask.shape[0] > 0:
            m = mask[i]
        else:
            m = None
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
            label = int(box[5])
        else:
            score = None
            label = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS[id % len(COLORS)]
        label_str = '{:d} {:s}'.format(id, CLASSES[label])
        img = plot_one_box([x1, y1, x2, y2], img, color, label_str, score=score, mask=m)
    return img