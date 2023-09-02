import cv2
import numpy as np

threshold = 0.45  # Threshold to detect object
input_size = 320, 320
nms_threshold = 0.2

cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)
# cap.set(10, 70)

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

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=threshold)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold)

    for i in indices:
        i = i[0] if type(i) is list or type(i) is np.ndarray else i

        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        class_id = classIds[i][0] if type(classIds[i]) is list or type(classIds[i]) is np.ndarray else classIds[i]
        cv2.putText(img, classNames[class_id - 1].upper(),
                    (np.max((0, np.min((input_size[0], box[0] + 10)))),
                     np.max((0, np.min((input_size[1], box[1] + 30))))),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # success,img = cap.read()
    # classIds, confs, bbox = net.detect(img,confThreshold=thres)
    # print(classIds,bbox)

    # if len(classIds) != 0:
    #     for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    #         cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    #         cv2.putText(img, classNames[classId - 1].upper(),
    #                     (np.max((0, np.min((input_size[0], box[0] + 10)))),
    #                      np.max((0, np.min((input_size[1], box[1] + 30))))),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    #         cv2.putText(img, str(round(confidence * 100, 2)),
    #                     (np.max((0, np.min((input_size[0], box[0] + 200)))),
    #                      np.max((0, np.min((input_size[1], box[1] + 30))))),
    #                     cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
