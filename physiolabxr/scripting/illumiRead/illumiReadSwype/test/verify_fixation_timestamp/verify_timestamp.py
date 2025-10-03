import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import cv2
import time
import os
import pickle
import sys
from physiolabxr.scripting.RenaScript import RenaScript
from pylsl import StreamInfo, StreamOutlet, cf_float32
import torch
from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeConfig import VarjoLSLChannelInfo
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeUtils import VarjoGazeData
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeFilterFixationDetectionIVT
from physiolabxr.utils.RNStream import RNStream

rn_stream = RNStream("11_15_2023_21_08_16-Exp_AnnaFixationTest-Sbj_Anna-Ssn_0.dats")
data = rn_stream.stream_in()

print(data)






# lsl_ts = data[GazeDataLSLStreamInfo.StreamName][1]
#
# # scatter plot lsl_ts x is the index of the sample, y is the timestamp
# # plt.scatter(range(len(lsl_ts[500:530])), lsl_ts[500:530])
#
#
#
#
#
# capture_ts = data[GazeDataLSLStreamInfo.StreamName][0][VarjoLSLChannelInfo.CaptureTime, :]
#
# print(capture_ts)


