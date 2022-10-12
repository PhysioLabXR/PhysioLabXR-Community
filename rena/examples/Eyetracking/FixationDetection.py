import time

import cv2
import zmq
import numpy as np
from pylsl import StreamInlet, resolve_stream

image_shape = (400, 400, 3)

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
        imagePNGBytes = cam_capture_sub_socket.recv_multipart()[1]
        img = cv2.imdecode(np.frombuffer(imagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)
        img_modified = img.copy()

        cv2.imshow('Camera Capture Object Detection', img)
        cv2.waitKey(delay=1)

        # get the most recent gaze tracking screen position
        sample, timestamp = inlet.pull_chunk()
        gaze_x, gaze_y = int(sample[-1][0]), int(sample[-1][1])  # the gaze coordinate
        gaze_y = image_shape[1] - gaze_y  # because CV's y zero is at the bottom of the screen
        center = gaze_x, gaze_y


    except KeyboardInterrupt:
        print('Stopped')

