import time

import cv2
import zmq
import numpy as np

image_shape = (480, 640, 3)

# parameters for object detection
threshold = 0.45  # Threshold to detect object
input_size = 320, 320
nms_threshold = 0.2
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(input_size)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# setup information for the server
subtopic = 'CamCapture1'
sub_tcpAddress = "tcp://localhost:5555"
rep_tcpAddress = 'tcp://*:5556'

context = zmq.Context()
cam_capture_sub_socket = context.socket(zmq.SUB)
cam_capture_sub_socket.connect(sub_tcpAddress)
cam_capture_sub_socket.setsockopt_string(zmq.SUBSCRIBE, subtopic)

od_req_socket = context.socket(zmq.REP)
od_req_socket.bind(rep_tcpAddress)

print('Sockets connected, entering image loop')

while True:
    try:
        imagePNGBytes = cam_capture_sub_socket.recv_multipart()[1]
        img = cv2.imdecode(np.frombuffer(imagePNGBytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(image_shape)

        classIds, confs, bbox = net.detect(img, confThreshold=threshold)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)
        detected_classes, xs, ys, ws, hs = list(), list(), list(), list(), list()
        for i in indices:
            i = i[0] if type(i) is list else i
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            xs.append(int(x + w/2))
            ys.append(int(y))
            ws.append(int(w))
            hs.append(int(h))

            class_id = classIds[i][0] if type(classIds[i]) is list else classIds[i]
            detected_classes.append(int(class_id))
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
            cv2.putText(img, classNames[class_id - 1].upper(),
                        (np.max((0, np.min((input_size[0], box[0] + 10)))),
                         np.max((0, np.min((input_size[1], box[1] + 30))))),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # response to Unity
        data = {
            'classIDs': detected_classes,
            'xs': xs,
            'ys': ys,
            'ws': ws,
            'hs': hs
        }
        od_req_socket.recv()
        od_req_socket.send_json(data)

        cv2.imshow('Camera Capture Object Detection', img)
        cv2.waitKey(delay=1)
    except KeyboardInterrupt:
        print('Stopped')

