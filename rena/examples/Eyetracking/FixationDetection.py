"""
Realtime fixation detection is always delayed <minimal fixation duration>
"""

import time

import cv2
import lpips
import zmq
import numpy as np
from pylsl import StreamInlet, resolve_stream, StreamOutlet, StreamInfo
from rena.examples.Eyetracking.EyeUtils import prepare_image_for_sim_score, add_bounding_box
from rena.examples.Eyetracking.configs import *


# fix detection parameters  #######################################
loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
previous_img_patch = None
fixation_frame_counter = 0

# LSL detected fixations ########################################
outlet = StreamOutlet(StreamInfo("FixationDetection", 'FixationDetection', 3, 30, 'float32'))

# zmq camera capture fields #######################################
subtopic = 'CamCapture1'
sub_tcpAddress = "tcp://localhost:5555"
context = zmq.Context()
cam_capture_sub_socket = context.socket(zmq.SUB)
cam_capture_sub_socket.connect(sub_tcpAddress)
cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, subtopic)

# LSL gaze screen position ########################################
streams = resolve_stream('name', 'Unity.gazeTargetScreenPosition')
inlet = StreamInlet(streams[0])


print('Sockets connected, entering image loop')

while True:
    try:
        fix_detection_sample = np.zeros(3) - 1

        imagePNGBytes = cam_capture_sub_socket.recv_multipart()[1]
        img = cv2.imdecode(np.frombuffer(imagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)
        img_modified = img.copy()

        # get the most recent gaze tracking screen position
        sample, timestamp = inlet.pull_chunk()
        if len(sample) < 1:
            continue
        gaze_x, gaze_y = int(sample[-1][0]), int(sample[-1][1])  # the gaze coordinate
        gaze_y = image_shape[1] - gaze_y  # because CV's y zero is at the bottom of the screen
        center = gaze_x, gaze_y

        img_patch = img[int(np.max([0, gaze_x - patch_size[0] / 2])) : int(np.min([image_size[0], gaze_x + patch_size[0] / 2])),
                    int(np.max([0, gaze_y - patch_size[1] / 2])):int(np.min([image_size[1], gaze_y + patch_size[1] / 2]))]

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
            fix_detection_sample[0] = distance
            fix_detection_sample[1] = fixation
            fix_detection_sample[2] = fixation_frame_counter >= fixation_min_frame_count
            outlet.push_sample(fix_detection_sample)

        previous_img_patch = img_patch

        img_modified = add_bounding_box(img_modified, gaze_x, gaze_y, patch_size[0], patch_size[1], patch_color)
        cv2.circle(img_modified, center, 1, center_color, 2)
        axis = (int(central_fov * ppds[0]), int(central_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, fovea_color, thickness=4)
        axis = (int(near_peripheral_fov * ppds[0]), int(near_peripheral_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, parafovea_color, thickness=4)
        axis = (int(1.25 * mid_perpheral_fov * ppds[0]), int(mid_perpheral_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, peripheri_color, thickness=4)

        cv2.imshow('Camera Capture Object Detection', img_modified)
        cv2.waitKey(delay=1)

    except KeyboardInterrupt:
        print('Stopped')

