import time
import warnings

import cv2
import zmq
import numpy as np
from os import path

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

target_class = "Motorcycle"
target_images = []
nontarget_images = []

image_shape = (480, 480, 3)

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
camera_capture_subscription_topic = 'CamCapture1'
subscribe_tcpAddress = "tcp://localhost:5556"
reply_tcpAddress = 'tcp://*:5557'


context = zmq.Context()
cam_capture_subscription_socket = context.socket(zmq.SUB)
cam_capture_subscription_socket.connect(subscribe_tcpAddress)
cam_capture_subscription_socket.setsockopt_string(zmq.SUBSCRIBE, camera_capture_subscription_topic)

object_detection_req_socket = context.socket(zmq.REP)
object_detection_req_socket.bind(reply_tcpAddress)

command_topic = "TargetIdentificationModelCommands"
command_subscribe_socket = context.socket(zmq.SUB)
command_subscribe_socket.connect(subscribe_tcpAddress)
command_subscribe_socket.setsockopt_string(zmq.SUBSCRIBE, command_topic)

print('Sockets connected, entering image loop')

# parameters for target detection in real time
imgs = []
labels = []
mode = "train"

def processDepthImage(depthROI):
    near = 0.1
    far = 20.0
    max16bitval = 65535;
    min16bitval = 0;
    filtered_depthROI = depthROI[depthROI != 0]
    # scale from 0-65535 to 0-1 value range
    scale = 1.0 / (max16bitval - min16bitval);
    compressed = filtered_depthROI * scale;
    # decompress values by 0.25 compression factor
    decompressed = np.power(compressed, 4)
    # remove non valid 0 depth values
    valid_decompressed = decompressed[decompressed != 0]
    # scale from eye to far rather than near to far (still linear 0-1 range)
    scaled_eye_far = -(valid_decompressed - 1) / (1 + near / far) + near / far
    scaled_eye_far = scaled_eye_far[scaled_eye_far != 0]

    # remove noisy outliers (there seem to be higher depth values than there should be)
    mean = np.mean(scaled_eye_far)
    standard_deviation = np.std(scaled_eye_far)
    distance_from_mean = abs(scaled_eye_far - mean)
    max_deviations = 2
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    no_outliers = scaled_eye_far[not_outlier]
    if len(no_outliers) == 0:
       return None
    return np.min(no_outliers), np.max(no_outliers), np.average(no_outliers)

def process_received_camera_images(received):
    depth_image_png_bytes = received[1]
    color_image_png_bytes = received[2]
    depthImg = cv2.imdecode(np.frombuffer(depth_image_png_bytes, dtype='uint8'), cv2.IMREAD_UNCHANGED)
    minDepth = []
    maxDepth = []
    aveDepth = []
    cv2.imshow('Depth Camera', depthImg)
    cv2.waitKey(delay=1)

    # Get color frame and perform 2D YOLO object detection
    colorImg = cv2.imdecode(np.frombuffer(color_image_png_bytes, dtype='uint8'), cv2.IMREAD_UNCHANGED).reshape(
        image_shape)

    classIds, confs, bbox = net.detect(colorImg, confThreshold=threshold)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)
    detected_classes, xs, ys, ws, hs = list(), list(), list(), list(), list()
    for i in indices:
        class_id = classIds[i][0] if type(classIds[i]) is list or type(classIds[i]) is np.ndarray else classIds[i]
        i = i[0] if type(i) is list or type(i) is np.ndarray else i
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        xs.append(int(x))
        ys.append(int(y))
        ws.append(int(w))
        hs.append(int(h))
        # Process depth information of bounding box region (min, max, depth)
        depthROI = depthImg[y:y + h, x:x + w]
        cv2.imshow('Depth ROI', depthROI)
        cv2.waitKey(delay=1)
        depth_results = processDepthImage(depthROI)
        if depth_results is None:
            warnings.warn(f"no valid depth point was found for class {classNames[class_id - 1]}")
            continue  # go to the next loop (next detected object)
        minD, maxD, aveD = depth_results
        # print("Min:", minD)
        # print("Max:", maxD)
        # print("Ave:", aveD)
        minDepth.append(minD)
        maxDepth.append(maxD)
        aveDepth.append(aveD)

        # add target and nontarget subimages to their dataset
        if classNames[class_id - 1] == target_class:
            target_images.append(colorImg[x:x + w, y:y + h])
        else:
            nontarget_images.append(colorImg[x:x + w, y:y + h])

        # Yolo 2D bb visualization
        detected_classes.append(int(class_id))
        cv2.rectangle(colorImg, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        cv2.putText(colorImg, classNames[class_id - 1].upper(),
                    (np.max((0, np.min((input_size[0], box[0] + 10)))),
                     np.max((0, np.min((input_size[1], box[1] + 30))))),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        # if classNames[class_id - 1] == 'motorcycle':
        #     # Get depth information of bounding box region

    # get dimensions of image
    dimensions = colorImg.shape

    # height, width, number of channels in image
    height = colorImg.shape[0]
    width = colorImg.shape[1]
    channels = colorImg.shape[2]

    # print('Image Dimension    : ', dimensions)
    # print('Image Height       : ', height)
    # print('Image Width        : ', width)
    # print('Number of Channels : ', channels)

    cv2.imshow('Camera Capture Object Detection', colorImg)
    cv2.waitKey(delay=1)

    # response to Unity
    data = {
        'classIDs': detected_classes,
        'xs': xs,
        'ys': ys,
        'ws': ws,
        'hs': hs,
        'minDepth': minDepth,
        'maxDepth': maxDepth,
        'aveDepth': aveDepth,
    }
    object_detection_req_socket.recv()
    object_detection_req_socket.send_json(data)

def process_command(received):
    print(f"received command {received}")


while True:
    try:
        # Get depth frame
        received_image = None
        while True:  # empty the buffers of zmq
            try:
                received_image = cam_capture_subscription_socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                break
        if received_image is not None:
            process_received_camera_images(received_image)

        received_command = None
        try:
            received_command = command_subscribe_socket.recv_multipart(flags=zmq.NOBLOCK)
        except zmq.error.Again:
            pass
        if received_command is not None:
            process_command(received_command)

    except KeyboardInterrupt:
        print('Stopped')

