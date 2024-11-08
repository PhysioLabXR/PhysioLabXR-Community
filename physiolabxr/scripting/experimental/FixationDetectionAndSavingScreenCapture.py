"""
Realtime fixation detection based on patch similarity.

Also decode the camera frames, both color and depth and save them to
"""
import json
import os.path
import time
from datetime import datetime

import cv2
import lpips
import pandas as pd
import zmq
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from physiolabxr.examples.Eyetracking.EyeUtils import prepare_image_for_sim_score, add_bounding_box, clip_bbox
from physiolabxr.examples.Eyetracking.configs import *
import struct



# fix detection parameters  #######################################
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
previous_img_patch = None
fixation_frame_counter = 0
distance = 0

# LSL detected fixations ########################################
outlet = StreamOutlet(StreamInfo("FixationDetection", 'FixationDetection', 3, 30, 'float32'))

# zmq camera capture fields #######################################
subtopic = 'ColorDepthCamGazePositionBBox'
sub_tcpAddress = "tcp://localhost:5556"
context = zmq.Context()
cam_capture_sub_socket = context.socket(zmq.SUB)
cam_capture_sub_socket.connect(sub_tcpAddress)
cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, subtopic)


# Disk Utilities Fields ########################################
capture_save_location = "C:/Users/LLINC-Lab/Documents/Recordings"
is_saving_captures = False
draw_fovea_on_image = False

now = datetime.now()
dt_string = now.strftime("%m_%d_%Y_%H_%M_%S")
capture_save_location = os.path.join(capture_save_location, 'ReNaUnityCameraCapture_' + dt_string)
os.makedirs(capture_save_location, exist_ok=False)
frame_counter = 0
df = pd.DataFrame(columns=['FrameNumber', 'GazePixelPositionX', 'GazePixelPositionY', 'LocalClock', 'bboxes'])
gaze_info_save_counter_max = 120  # frames
gaze_info_save_counter = 0
gaze_info_path = os.path.join(capture_save_location, 'GazeInfo.csv')

print(f'Sockets connected, entering image loop. Writing to {capture_save_location}')
while True:
    try:
        fix_detection_sample = np.zeros(3) - 1

        received = cam_capture_sub_socket.recv_multipart()
        timestamp = struct.unpack('d', received[1])[0]
        colorImagePNGBytes = received[2]
        depthImagePNGBytes = received[3]
        # colorImg = cv2.imdecode(np.frombuffer(colorImagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)
        colorImg = np.frombuffer(colorImagePNGBytes, dtype='uint8').reshape((*image_shape[:2], 4))
        # remove the alpha channel
        colorImg = colorImg[:, :, :3]
        colorImg = cv2.flip(colorImg, 0)
        colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)

        # convert from bgr to rgb
        # depthImg = cv2.imdecode(np.frombuffer(depthImagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape((*image_shape[:2], 1))
        depthImg = np.frombuffer(depthImagePNGBytes, dtype='uint16').reshape((*image_shape[:2], 1))
        depthImg = cv2.flip(depthImg, 0)

        gaze_info = received[5]
        gaze_x, gaze_y = struct.unpack('hh', gaze_info)  # the gaze coordinate
        gaze_y = image_shape[1] - gaze_y  # because CV's y zero is at the bottom of the screen

        # get the item bboxes
        item_bboxes = received[4]
        item_bboxes = json.loads(item_bboxes)  # decode item bbox as json

        # save the original image
        if is_saving_captures:
            cv2.imwrite(os.path.join(capture_save_location, '{}.png'.format(frame_counter)), colorImg)
            cv2.imwrite(os.path.join(capture_save_location, '{}_depth.png'.format(frame_counter)), depthImg)
        frame_counter += 1

        # write to gaze info csv
        row = {'FrameNumber': frame_counter, 'GazePixelPositionX': gaze_x, 'GazePixelPositionY': gaze_y, 'LocalClock': timestamp, 'bboxes': json.dumps(item_bboxes)}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        gaze_info_save_counter += 1
        if gaze_info_save_counter >= gaze_info_save_counter_max:
            df.to_csv(gaze_info_path)
            gaze_info_save_counter = 0

        # get all available item markers
        img_modified = colorImg.copy()

        # get the most recent gaze tracking screen position
        # sample, timestamp = inlet.pull_chunk()
        # if len(sample) < 1:
        #     continue
        center = gaze_x, gaze_y

        img_patch_x_min = int(np.min([np.max([0, gaze_x - patch_size[0] / 2]), image_size[0] - patch_size[0]]))
        img_patch_x_max = int(np.max([np.min([image_size[0], gaze_x + patch_size[0] / 2]), patch_size[0]]))
        img_patch_y_min = int(np.min([np.max([0, gaze_y - patch_size[1] / 2]), image_size[1] - patch_size[1]]))
        img_patch_y_max = int(np.max([np.min([image_size[1], gaze_y + patch_size[1] / 2]), patch_size[1]]))
        img_patch = colorImg[img_patch_x_min : img_patch_x_max,
                         img_patch_y_min: img_patch_y_max]
        patch_boundary = (img_patch_x_min, img_patch_y_min, img_patch_x_max, img_patch_y_max)

        if previous_img_patch is not None:
            img_tensor, previous_img_tensor = prepare_image_for_sim_score(img_patch), prepare_image_for_sim_score(
                previous_img_patch)
            distance = loss_fn_alex(img_tensor, previous_img_tensor).item()
            fixation = 0 if distance > similarity_threshold else 1
            if fixation == 0:
                fixation_frame_counter = 0
            else:
                fixation_frame_counter += 1
            # add to LSL
            fix_detection_sample[0] = 1 - distance
            fix_detection_sample[1] = fixation
            fix_detection_sample[2] = fixation_frame_counter >= fixation_min_frame_count
            outlet.push_sample(fix_detection_sample)

        previous_img_patch = img_patch

        # bounding rectange for the central patch
        shapes = np.zeros_like(img_modified, np.uint8)
        alpha = (1 - distance) / 2
        cv2.rectangle(shapes, patch_boundary[:2], patch_boundary[2:], patch_color, thickness=-1)
        mask = shapes.astype(bool)
        img_modified[mask] = cv2.addWeighted(img_modified, alpha, shapes, 1 - alpha, 0)[mask]

        cv2.circle(img_modified, center, 1, center_color, 2)
        axis = (int(central_fov * ppds[0]), int(central_fov * ppds[1]))

        if draw_fovea_on_image:
            cv2.ellipse(img_modified, center, axis, 0, 0, 360, fovea_color, thickness=4)
            axis = (int(near_peripheral_fov * ppds[0]), int(near_peripheral_fov * ppds[1]))
            cv2.ellipse(img_modified, center, axis, 0, 0, 360, parafovea_color, thickness=4)
            axis = (int(1.25 * mid_perpheral_fov * ppds[0]), int(mid_perpheral_fov * ppds[1]))
            cv2.ellipse(img_modified, center, axis, 0, 0, 360, peripheri_color, thickness=4)

        # put the item bboxes on the image
        for item_index, item_bbox in item_bboxes.items():
            # clip the bbox to the image size, the bbox is in the format x_center, y_center, width, height
            item_bbox_clipped = clip_bbox(*item_bbox, image_shape)

            img_modified = add_bounding_box(img_modified, *item_bbox_clipped, color=(0, 255, 0))
            # put the item index as text
            cv2.putText(img_modified, str(item_index), (item_bbox_clipped[0], item_bbox_clipped[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # convert from bgr to rgb for cv2 display, also need to on the vertical axis

        cv2.imshow('Camera Capture Object Detection', img_modified)
        cv2.waitKey(delay=1)

        # display the depth image
        cv2.imshow('Depth Image', depthImg)
        cv2.waitKey(delay=1)

    except KeyboardInterrupt:
        print('Stopped')

