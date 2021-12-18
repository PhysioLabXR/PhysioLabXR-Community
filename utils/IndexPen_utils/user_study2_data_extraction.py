import os
import numpy as np
from utils.IndexPen_utils.preprocessing_utils import load_idp, load_idp_file, load_idp_raw
import pickle

# user_study2_data_dir = 'Z:/hwei/IndexPen_Data/user_study2_data'
user_study2_data_dir = 'C:/Recordings/user_study2_data'
user_study2_data_save_dir = 'C:/Users/Haowe/PycharmProjects/IndexPen_Training/data/IndexPenData/IndexPenStudyData/UserStudy2Data'
# user_study2_data_save_dir = 'D:/IndexPen_data_Study1_Study2/user_study2_data/'
participant_dir = 'participant_3'
session_dir = 'session_5'

full_session_dir_path = os.path.join(user_study2_data_dir, participant_dir, session_dir)

exp_info_dict_json_path = '../../utils/IndexPen_utils/IndexPenExp.json'
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
DataStreamName = 'TImmWave_6843AOP'

fs = 30
duration = 4
sample_num = fs * duration

session_data_dict = {}
session_label_dict = {}

trails_name = os.listdir(full_session_dir_path)
for trail_index, trail_name in enumerate(trails_name):
    file_name_info = os. path. splitext(trail_name)
    trail_session_index = int(file_name_info[0].split('_')[-1])
    trail_data_dict = {}

    trail_path = os.path.join(full_session_dir_path, trail_name)

    indexpen_train = load_idp_file(trail_path, DataStreamName, reshape_dict, exp_info_dict_json_path,
                                   sample_num, rd_cr_ratio=0.8, ra_cr_ratio=0.8, all_categories=None)

    indexpen_raw = load_idp_raw(
        data_file_path=trail_path,
        DataStreamName=DataStreamName,
        reshape_dict=reshape_dict,
        exp_info_dict_json_path=exp_info_dict_json_path,
        rd_cr_ratio=0.8,
        ra_cr_ratio=0.8,
        all_categories=None,
        session_only=True
    )

    session_data_dict[trail_session_index] = [indexpen_train, indexpen_raw]

# # all data extraction done
#
with open(os.path.join(user_study2_data_save_dir, participant_dir, session_dir), 'wb') as f:
    pickle.dump([participant_dir, session_dir, session_data_dict], f, protocol=4)

# all_data_dir_path = 'C:/Recordings/transfer_learning_test/transfer_learning_test_rnstream'
# for root, subdirs, files in os.walk(all_data_dir_path):
#     print(root)
#     print(subdirs)
#     print(files)
