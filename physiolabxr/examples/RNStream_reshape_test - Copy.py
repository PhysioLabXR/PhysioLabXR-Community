import numpy
import numpy as np

from utils.data_utils import RNStream

# test_rns = RNStream('C:/Recordings/05_27_2021_18_46_38-Exp_myexperiment-Sbj_john-Ssn_0.dats')
# test_reloaded_data = test_rns.stream_in()

# shapes = [(8, 16, 1), (8, 64, 1), (640,)]
#
# # reshape RN stream split frame
# data_frame = test_reloaded_data['TImmWave_6843AOP'][0]
#
# offset = 0
# channel1_num = numpy.prod(shapes[0])
# channel1 = data_frame[offset: channel1_num, :]
# test = channel1[:, 0]
# channel1 = channel1.reshape((8, 16, 1, -1))
# watch = channel1[:, :, :, 310]
# watch = np.squeeze(watch, axis=-1)
#
# print(numpy.prod(shapes[2]))
#
#
# reshape_channel_num = 0
# print(test_reloaded_data['TImmWave_6843AOP'][0].shape[-1])
# print(shapes[2][0])


test_rns = RNStream('C:/Recordings/05_27_2021_18_46_38-Exp_myexperiment-Sbj_john-Ssn_0.dats')
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
                }

test_reloaded_data = test_rns.stream_in(reshape_stream_dict=reshape_dict)
