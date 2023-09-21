import warnings

import cv2
import zmq
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript


def get_od_model(config_path, weights_path, input_size):
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(input_size)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net


def get_class_names(class_file):
    with open(class_file, 'rt') as f:
        class_name = f.read().rstrip('\n').split('\n')
    return class_name


def process_depth_image(depthROI):
    near = 0.1
    far = 20.0
    max16bitval = 65535
    min16bitval = 0
    filtered_depthROI = depthROI[depthROI != 0]
    # scale from 0-65535 to 0-1 value range
    scale = 1.0 / (max16bitval - min16bitval)
    compressed = filtered_depthROI * scale
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


def process_received_camera_images(image_data, net, class_names, image_shape, threshold=0.45, nms_threshold=0.2):
    depth_img = depth_image_data.reshape(image_shape).astype(np.uint8)
    color_img = image_data.reshape(image_shape).astype(np.uint8)

    classIds, confs, bbox = net.detect(color_img, confThreshold=threshold)
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

        # Yolo 2D bb visualization
        detected_classes.append(int(class_id))
        cv2.rectangle(color_img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        cv2.putText(color_img, class_names[class_id - 1].upper(),
                    (np.max((0, np.min((image_shape[0], box[0] + 10)))),
                     np.max((0, np.min((image_shape[1], box[1] + 30))))),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return {
        'classIDs': detected_classes,
        'xs': xs,
        'ys': ys,
        'ws': ws,
        'hs': hs,
    }, color_img

class BaseRenaScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        config_path = 'physiolabxr/examples/ObjectDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights_path = 'physiolabxr/examples/ObjectDetection/frozen_inference_graph.pb'
        self.image_shape = (400, 400, 3)
        self.ob_model = get_od_model(config_path, weights_path, input_size=self.image_shape[:2])
        self.class_names = get_class_names('physiolabxr/examples/ObjectDetection/coco.names')


    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        if "CamCapture" in self.inputs and "DepthCamCapture" in self.inputs:
            image_data = self.inputs["CamCapture"][0][:, -1]
            detected_pos, img_w_bbx = process_received_camera_images(image_data, self.ob_model, self.class_names, self.image_shape)
            self.outputs["OutputImg"] = img_w_bbx.reshape(-1)
            self.inputs.clear_buffer()


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
