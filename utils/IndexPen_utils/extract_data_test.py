import numpy as np
import json
from utils.data_utils import RNStream

test_rns = RNStream('C:/Recordings/05_28_2021_02_41_44-Exp_IndexPen-Sbj_test_exp-Ssn_0.dats')
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}

data_ts = test_rns.stream_in(reshape_stream_dict=reshape_dict)

# extract experiment session

# extract 120 frames after each label event marker
sample_num = 120

exp_info_dict = json.load(open('../IndexPen_utils/IndexPenExp.json'))

# get useful timestamps
ExpID = exp_info_dict['ExpID']
ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
ExpStartMarker = exp_info_dict['ExpStartMarker']
ExpEndMarker = exp_info_dict['ExpEndMarker']
ExpLabelMarker = exp_info_dict['ExpLabelMarker']
ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
ExpErrorMarker = exp_info_dict['ExpErrorMarker']

index_buffer = []
label_buffer = []

event_markers = data_ts[ExpLSLStreamName][0][0]
start_marker_indexes = np.where(event_markers == 100)[0]

for start_marker_index in start_marker_indexes:
    # forward track the event marker
    session_index_buffer = []
    session_label_buffer = []
    for index in range(start_marker_index+1, len(event_markers)):
        # stop the forward tracking and go for the next session if interrupt Marker found
        if event_markers[index] == ExpInterruptMarker:
            break
        elif event_markers[index] == ExpID:
            break
        elif event_markers[index] == ExpEndMarker:
            # only attach the event marker with regular exit
            index_buffer.extend(session_index_buffer)
            label_buffer.extend(session_label_buffer)
            break

        # remove last element from the list
        if event_markers[index] == ExpErrorMarker and len(session_index_buffer) != 0:
            del session_index_buffer[-1]
            del session_index_buffer[-1]

        session_index_buffer.append(index)
        session_label_buffer.append(event_markers[index])

# get all timestamps using index list
label_time_stamps = data_ts[ExpLSLStreamName][1][index_buffer]


