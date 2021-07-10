import pickle

from preprocessing_utils import load_idp

DataStreamName = 'TImmWave_6843AOP'
# data_dir_path = 'C:\Recordings\John_F-J'
data_dir_path = 'C:/Recordings/transfer_learning_test/transfer_learning_test_rnstream/John/new_gesture'
# data_dir_path = 'C:/Recordings/John_A-J_test/2'

exp_info_dict_json_path = 'C:/Users/Haowe/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'
save_data_dir = 'C:/Recordings/transfer_learning_test/transfer_learning_test_pickle/John/John_20_new_gesture_sample_clutter_removal_(0.8)_(0.6)_transfer_learning_test'
# save_data_dir = 'C:/Recordings/John_A-J_test/2/C-G_test'

reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
fs = 30
duration = 4
sample_num = fs * duration
# categories = [1, 2, 3, 4, 5]
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_dict, Y, encoder = load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num, all_categories=None)

with open(save_data_dir, 'wb') as f:
    pickle.dump([X_dict, Y, encoder], f)