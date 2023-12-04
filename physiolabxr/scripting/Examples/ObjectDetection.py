import warnings

import cv2
import zmq
import numpy as np
from physiolabxr.scripting.RenaScript import RenaScript

# get the object detection model
def get_od_model(config_path, weights_path, input_size):
    net = cv2.dnn_DetectionModel(weights_path, config_path)
    net.setInputSize(input_size)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return net

# get the class names from the coco.names file for the object detection model
def get_class_names(class_file):
    with open(class_file, 'rt') as f:
        class_name = f.read().rstrip('\n').split('\n')
    return class_name


# process the received camera images
def process_received_camera_images(image_data, net, class_names, image_shape, threshold=0.45, nms_threshold=0.2):
    color_img = image_data.reshape(image_shape).astype(np.uint8) # reshape the image data to the image shape
    color_img = cv2.rotate(color_img, cv2.ROTATE_90_COUNTERCLOCKWISE) # rotate the image 90 degrees counter clockwise because the cv2 has a different origin

    classIds, confs, bbox = net.detect(color_img, confThreshold=threshold) # get the bounding boxes, confidence, and class ids
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, threshold, nms_threshold) # get the indices of the bounding boxes
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

    color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE) # rotate the image back to its original orientation

    # return the detected classes, the positions, and the image with bounding boxes
    return {
        'classIDs': detected_classes,
        'xs': xs,
        'ys': ys,
        'ws': ws,
        'hs': hs,
    }, color_img

class ObjectDetectionScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        config_path = 'physiolabxr/examples/ObjectDetection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weights_path = 'physiolabxr/examples/ObjectDetection/frozen_inference_graph.pb'
        self.image_shape = (640, 480, 3)
        self.ob_model = get_od_model(config_path, weights_path, input_size=self.image_shape[:2])
        self.class_names = get_class_names('physiolabxr/examples/ObjectDetection/coco.names')


    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        camera_stream_names = [x for x in self.inputs.keys() if x.startswith("Camera")]
        if len(camera_stream_names) > 0: # check if the camera is in the inputs
            # take the first stream whose name starts with camera
            stream_name = camera_stream_names[0]
            image_data = self.inputs[stream_name][0][:, -1] # get the newest image data from the camera
            detected_pos, img_w_bbx = process_received_camera_images(image_data, self.ob_model, self.class_names, self.image_shape) # process the image data
            self.outputs["OutputImg"] = img_w_bbx.reshape(-1) # reshape the output image to send
            self.inputs.clear_buffer() # clear the input buffer



    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
