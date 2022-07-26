from utils.data_utils import RNStream

data_file_path = 'C:/Recordings/07_26_2022_00_32_27-Exp_nearfoward-Sbj_someone-Ssn_0.dats'
rs_stream = RNStream(data_file_path)

reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)],
    'TImmWave_6843AOP_CR': [(8, 16, 1), (8, 64, 1)]
}

data = rs_stream.stream_in(reshape_stream_dict=reshape_dict, ignore_stream=None, jitter_removal=False)
