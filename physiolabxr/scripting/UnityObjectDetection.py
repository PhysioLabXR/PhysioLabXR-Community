import warnings

import cv2
import zmq
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface


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

def processDepthImage(depthROI):
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

def process_received_camera_images(image_data, image_depth_data, net, class_names, image_shape, threshold=0.45, nms_threshold=0.2):
    color_img = image_data.reshape(image_shape).astype(np.uint8)
    image_depth = image_depth_data.reshape(image_shape[:2])

    minDepth = []
    maxDepth = []
    aveDepth = []

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

        # Process depth information of bounding box region (min, max, depth)
        depthROI = image_depth[y:y + h, x:x + w]

        depth_results = processDepthImage(depthROI)
        if depth_results is None:
            # warnings.warn(f"no valid depth point was found for class {classNames[class_id - 1]}")
            continue  # go to the next loop (next detected object)
        minD, maxD, aveD = depth_results
        minDepth.append(minD)
        maxDepth.append(maxD)
        aveDepth.append(aveD)

        # Yolo 2D bb visualization
        detected_classes.append(int(class_id))
        cv2.rectangle(color_img, (x, y), (x + w, h + y), color=(0, 255, 0), thickness=2)
        cv2.putText(color_img, class_names[class_id - 1].upper(),
                    (np.max((0, np.min((image_shape[0], box[0] + 10)))),
                     np.max((0, np.min((image_shape[1], box[1] + 30))))),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, bottomLeftOrigin=True)

    color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # preprocess the image
    color_img = cv2.flip(color_img, 0)  # preprocess the image

    image_depth = cv2.rotate(image_depth, cv2.ROTATE_90_COUNTERCLOCKWISE)  # preprocess the image
    image_depth = cv2.flip(image_depth, 0)
    image_depth = image_depth.reshape(-1)

    ys = [image_shape[0] - y - h for y, h in zip(ys, hs)]  # flip the y coordinates

    return {
        'classIDs': detected_classes,
        'xs': xs,
        'ys': ys,
        'ws': ws,
        'hs': hs,
        'minDepth': minDepth,
        'maxDepth': maxDepth,
        'aveDepth': aveDepth,
    }, color_img, image_depth

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

        self.object_detection_rep_socket = RenaTCPInterface(stream_name='od_rep',
                                                            port_id='5557',
                                                            identity='server',
                                                            pattern='request-reply')

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        if "CamCapture" in self.inputs:
            image_data = self.inputs["CamCapture"][0][:, -1]
            n_channels = 400 * 400 * 3
            image_color = image_data[:n_channels]
            image_depth = np.frombuffer(image_data[n_channels:].tobytes(), dtype=np.uint16)

            detected_pos, img_w_bbx, img_depth = process_received_camera_images(image_color, image_depth, self.ob_model, self.class_names, self.image_shape, threshold=self.params['conf_threshold'])

            self.outputs["OutputImg"] = img_w_bbx.reshape(-1)
            self.outputs["DepthImg"] = img_depth
            self.inputs.clear_buffer()

            try:
                self.object_detection_rep_socket.socket.recv(flags=zmq.NOBLOCK)
                self.object_detection_rep_socket.socket.send_json(detected_pos)
            except zmq.error.Again:
                pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
