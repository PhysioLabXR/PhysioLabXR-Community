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
import pandas as pd
import zmq

from physiolabxr.examples.Eyetracking.EyeUtils import add_bounding_box, clip_bbox
from physiolabxr.examples.Eyetracking.configs import *


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
    item_bboxes = received[4]

    colorImg = np.frombuffer(colorImagePNGBytes, dtype='uint8').reshape((*image_shape[:2], 4))
    depthImg = np.frombuffer(depthImagePNGBytes, dtype='uint16').reshape((*image_shape[:2], 1))

    colorImg = colorImg[:, :, :3]
    colorImg = cv2.flip(colorImg, 0)
    colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)
    depthImg = cv2.flip(depthImg, 0)

    item_bboxes = json.loads(item_bboxes)

    return colorImg, depthImg, timestamp, item_bboxes


if __name__ == '__main__':
    is_saving_captures = True
    is_displaying = True

    right_cam_socket = get_cam_socket("tcp://localhost:5557", 'ColorDepthCamRight')
    left_cam_socket = get_cam_socket("tcp://localhost:5558", 'ColorDepthCamLeft')
    back_cam_socket = get_cam_socket("tcp://localhost:5559", 'ColorDepthCamBack')


    # Disk Utilities Fields ########################################
    capture_save_location = "C:/Recordings"

    now = datetime.now()
    dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
    capture_save_location = os.path.join(capture_save_location, 'ReNaUnityCameraCapture_' + dt_string)

    capture_right_path = os.path.join(capture_save_location, 'RightCamCapture')
    capture_left_path = os.path.join(capture_save_location, 'LeftCamCapture')
    capture_back_path = os.path.join(capture_save_location, 'BackCamCapture')

    csv_path_right = os.path.join(capture_right_path, 'GazeInfo.csv')
    csv_path_left = os.path.join(capture_left_path, 'GazeInfo.csv')
    csv_path_back = os.path.join(capture_back_path, 'GazeInfo.csv')

    csv_save_counter_max = 4  # frames
    csv_save_counter = 0  # frames

    if is_saving_captures:
        print(f'Writing to {capture_save_location}')
        os.makedirs(capture_right_path, exist_ok=False)
        os.makedirs(capture_left_path, exist_ok=False)
        os.makedirs(capture_back_path, exist_ok=False)

        for csv_path in [csv_path_right, csv_path_left, csv_path_back]:
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=['FrameNumber', 'LocalClock', 'bboxes'])
                writer.writeheader()

    frame_counter = 0

    print('Sockets connected, entering image loop')
    while True:
        try:
            fix_detection_sample = np.zeros(3) - 1

            right_cam_color, right_cam_depth, timestamp_right, bboxes_right = receive_decode_image(right_cam_socket)
            left_cam_color, left_cam_depth, timestamp_left, bboxes_left = receive_decode_image(left_cam_socket)
            back_cam_color, back_cam_depth, timestamp_back, bboxes_back = receive_decode_image(back_cam_socket)

            # save the original image
            if is_saving_captures:
                cv2.imwrite(os.path.join(capture_right_path, f"{frame_counter}.png"), right_cam_color)
                cv2.imwrite(os.path.join(capture_right_path, f"{frame_counter}_depth.png"), right_cam_depth)

                cv2.imwrite(os.path.join(capture_left_path, f"{frame_counter}.png"), left_cam_color)
                cv2.imwrite(os.path.join(capture_left_path, f"{frame_counter}_depth.png"), left_cam_depth)

                cv2.imwrite(os.path.join(capture_back_path, f"{frame_counter}.png"), back_cam_color)
                cv2.imwrite(os.path.join(capture_back_path, f"{frame_counter}_depth.png"), back_cam_depth)

                with open(csv_path_right, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['FrameNumber', 'LocalClock', 'bboxes'])
                    writer.writerow({'FrameNumber': frame_counter, 'LocalClock': timestamp_right, 'bboxes': json.dumps(bboxes_right)})

                with open(csv_path_left, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['FrameNumber', 'LocalClock', 'bboxes'])
                    writer.writerow({'FrameNumber': frame_counter, 'LocalClock': timestamp_left, 'bboxes': json.dumps(bboxes_left)})

                with open(csv_path_back, mode='a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['FrameNumber', 'LocalClock', 'bboxes'])
                    writer.writerow({'FrameNumber': frame_counter, 'LocalClock': timestamp_back, 'bboxes': json.dumps(bboxes_back)})

            frame_counter += 1

            #---------------------------------------------------------------------------------------------
            if is_displaying:
                # get all available item markers
                img_modified_left = left_cam_color.copy()
                img_modified_right = right_cam_color.copy()
                img_modified_back = back_cam_color.copy()

                # put the item bboxes on the image
                # ----------------------------------------------------------------------------------------------
                for item_index, item_bbox in bboxes_left.items():
                    item_bbox_clipped = clip_bbox(*item_bbox, image_shape)
                    img_modified_left = add_bounding_box(img_modified_left, *item_bbox_clipped, color=(0, 255, 0))
                    cv2.putText(img_modified_left, str(item_index), (item_bbox_clipped[0], item_bbox_clipped[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                for item_index, item_bbox in bboxes_right.items():
                    item_bbox_clipped = clip_bbox(*item_bbox, image_shape)
                    img_modified_right = add_bounding_box(img_modified_right, *item_bbox_clipped, color=(0, 255, 0))
                    cv2.putText(img_modified_right, str(item_index), (item_bbox_clipped[0], item_bbox_clipped[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                for item_index, item_bbox in bboxes_back.items():
                    item_bbox_clipped = clip_bbox(*item_bbox, image_shape)
                    img_modified_back = add_bounding_box(img_modified_back, *item_bbox_clipped, color=(0, 255, 0))
                    cv2.putText(img_modified_back, str(item_index), (item_bbox_clipped[0], item_bbox_clipped[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

                # displaying
                cv2.imshow('Right', img_modified_right)
                cv2.imshow('RightDepth', right_cam_depth)

                cv2.imshow('Left', img_modified_left)
                cv2.imshow('LeftDepth', left_cam_depth)

                cv2.imshow('Back', img_modified_back)
                cv2.imshow('BackDepth', back_cam_depth)

                cv2.waitKey(delay=1)

        except KeyboardInterrupt:
            print('Stopped')

