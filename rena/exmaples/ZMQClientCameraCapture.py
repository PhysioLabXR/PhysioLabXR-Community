import time

import cv2
import zmq
import numpy as np

image_shape = (720, 1280, 3)
subtopic = 'CamCapture1'
tcpAddress = "tcp://localhost:5555"

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(tcpAddress)
socket.setsockopt_string(zmq.SUBSCRIBE, subtopic)

print('Socket connected, entering image loop')

while True:

    imagePNGBytes = socket.recv_multipart()[1]
    frame = cv2.imdecode(np.frombuffer(imagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)

    cv2.imshow('Camera Capture', frame)
    cv2.waitKey(delay=1)

