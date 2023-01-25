import time

import cv2
import zmq

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind('tcp://*:5556')

capture = cv2.VideoCapture(0)
print('Video device connected, entering image loop')

while True:
    start = time.time()

    _, frame = capture.read()
    socket.send(cv2.cvtColor(cv2.flip(frame, 0), cv2.COLOR_BGR2RGB).tobytes())

    end = time.time()
    print('FPS:', 1 / (end - start))

    cv2.imshow('Webcam', frame)
    cv2.waitKey(delay=1)