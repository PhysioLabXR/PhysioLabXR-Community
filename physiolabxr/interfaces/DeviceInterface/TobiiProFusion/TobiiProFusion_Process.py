import zmq
import time
from ctypes import *
tobii = CDLL('physiolabxr/thirdparty/TobiiProSDKMac/64/lib/libtobii_research.dylib')


class TobiiResearchPoint3D(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("z", c_float)]

class TobiiResearchNormalizedPoint2D(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float)]

class TobiiResearchGazePoint(Structure):
    _fields_ = [("position_on_display_area", TobiiResearchNormalizedPoint2D),
                ("position_in_user_coordinates", TobiiResearchPoint3D),
                ("validity", c_int)]

class TobiiResearchPupilData(Structure):
    _fields_ = [("diameter", c_float),
                ("validity", c_int)]

class TobiiResearchGazeOrigin(Structure):
    _fields_ = [("position_in_user_coordinates", TobiiResearchPoint3D),
                ("validity", c_int)]

class TobiiResearchEyeData(Structure):
    _fields_ = [("gaze_point", TobiiResearchGazePoint),
                ("pupil_data", TobiiResearchPupilData),
                ("gaze_origin", TobiiResearchGazeOrigin)]

class TobiiResearchGazeData(Structure):
    _fields_ = [("left_eye", TobiiResearchEyeData),
                ("right_eye", TobiiResearchEyeData),
                ("device_time_stamp", c_int64),
                ("system_time_stamp", c_int64)]

@CFUNCTYPE(None, POINTER(TobiiResearchGazeData), c_void_p)
def gaze_data_callback(gaze_data, user_data):
    system_timestamp = gaze_data.contents.system_time_stamp
    device_timestamp = gaze_data.contents.device_time_stamp

    left_eye_gaze_origin = gaze_data.contents.left_eye.gaze_origin.position_in_user_coordinates
    right_eye_gaze_origin = gaze_data.contents.right_eye.gaze_origin.position_in_user_coordinates

    left_eye_gaze_point = gaze_data.contents.left_eye.gaze_point.position_on_display_area
    right_eye_gaze_point = gaze_data.contents.right_eye.gaze_point.position_on_display_area

    left_pupil_diameter = gaze_data.contents.left_eye.pupil_data.diameter
    right_pupil_diameter = gaze_data.contents.right_eye.pupil_data.diameter


    data = {
        "timestamp": system_timestamp,
        "device_timestamp": device_timestamp,
        "left_eye": {
            "gaze_origin": {"x": left_eye_gaze_origin.x, "y": left_eye_gaze_origin.y, "z": left_eye_gaze_origin.z},
            "gaze_point": {"x": left_eye_gaze_point.x, "y": left_eye_gaze_point.y},
            "pupil_diameter": left_pupil_diameter
        },
        "right_eye": {
            "gaze_origin": {"x": right_eye_gaze_origin.x, "y": right_eye_gaze_origin.y, "z": right_eye_gaze_origin.z},
            "gaze_point": {"x": right_eye_gaze_point.x, "y": right_eye_gaze_point.y},
            "pupil_diameter": right_pupil_diameter
        }
    }

    tobiiprofusion_data_socket.send_json(data)

@CFUNCTYPE(None, POINTER(c_int), c_void_p)
def buffer_overflow_notification_callback(notification, user_data):
    print("Buffer overflow notification received")


def TobiiProFusion_process(terminate_event, port):
    global tobiiprofusion_data_socket

    context = zmq.Context()
    tobiiprofusion_data_socket = context.socket(zmq.PUSH)
    tobiiprofusion_data_socket.bind(f"tcp://*:{port}")

    eyetracker = POINTER(TobiiResearchGazeData)()
    result = tobii.tobii_research_find_all_eyetrackers(byref(eyetracker))

    if result != 0 or not eyetracker:
        print("Failed to find eye tracker")
        return

    status = tobii.tobii_research_subscribe_to_gaze_data(eyetracker, gaze_data_callback, None)
    if status != 0:
        print("Failed to subscribe to gaze data")
        return

    tobii.tobii_research_subscribe_to_notifications(eyetracker, buffer_overflow_notification_callback, None)

    while not terminate_event.is_set():
        time.sleep(0.01)

    tobii.tobii_research_unsubscribe_from_gaze_data(eyetracker, gaze_data_callback)
    tobii.tobii_research_unsubscribe_from_notifications(eyetracker, buffer_overflow_notification_callback)

    tobiiprofusion_data_socket.close()
    context.term()
    print("TobiiProFusion process stopped.")
