from utils.IndexPen_utils.preprocessing_utils import load_idp

DataStreamName = 'TImmWave_6843AOP'
data_dir_path = 'C:/Users/HaowenWeiJohn/Desktop/IndexPen_Data/test_data'
exp_info_dict_json_path = 'C:/Users/HaowenWeiJohn/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
fs = 30
duration = 4
sample_num = fs * duration

X_dict, Y, encoder = load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num)

