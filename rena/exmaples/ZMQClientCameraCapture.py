import time

import cv2
import zmq
import numpy as np

image_shape = (720, 1280, 3)

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind('tcp://*:5555')

print('Socket connected, entering image loop')

while True:

    imagePNGBytes = socket.recv()
    frame = cv2.imdecode(np.frombuffer(imagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)

    cv2.imshow('Camera Capture', frame)
    cv2.waitKey(delay=1)

