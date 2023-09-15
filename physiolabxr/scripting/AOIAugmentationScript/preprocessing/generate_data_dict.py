import time
from collections import deque

from pylsl import StreamInfo, StreamOutlet, cf_float32

from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, GazeFilterFixationDetectionIVT, \
    tobii_gaze_on_display_area_to_image_matrix_index, GazeType, gaze_point_on_image_valid
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import EventMarkerLSLStreamInfo, GazeDataLSLStreamInfo
import torch




# data dict structure:
# data_dict = {
#     'image_path': image_path,
#     'image_rgb': image_rgb, (0-1)
#     'y': label, (G or S)
#     'y_pred': prediction, (G or S)
#     'attention_matrix_raw': attention_matrix_raw, (0-1)
#     'attention_matrix': attention_matrix, (0-1)
#}

class ImageAttentionClass():
    image_path = None







