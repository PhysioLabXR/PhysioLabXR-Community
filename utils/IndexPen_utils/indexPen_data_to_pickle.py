import pickle

from preprocessing_utils import load_idp

DataStreamName = 'TImmWave_6843AOP'
# data_dir_path = 'C:\Recordings\John_F-J'
data_dir_path = 'C:\Recordings\John_data'

exp_info_dict_json_path = 'C:/Users/HaowenWeiJohn/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'
save_data_dir = 'C:/Recordings/IndexPen_pickle/Haowen/test_data/test_A-J'

reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
fs = 30
duration = 4
sample_num = fs * duration
# categories = [1, 2, 3, 4, 5]
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_dict, Y, encoder = load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num, all_categories=categories)

with open(save_data_dir, 'wb') as f:
    pickle.dump([X_dict, Y, encoder], f)