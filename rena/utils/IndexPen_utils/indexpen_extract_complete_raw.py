import os
import numpy as np
from utils.IndexPen_utils.preprocessing_utils import load_idp, load_idp_file, load_idp_raw
import pickle

# user_study2_data_dir = 'Z:/hwei/IndexPen_Data/user_study2_data'
# user_study2_data_dir = 'C:/Recordings/user_study2_data'
# user_study2_data_save_dir = 'C:/Users/Haowe/PycharmProjects/IndexPen_Training/data/IndexPenData/IndexPenStudyData/UserStudy2Data'
# user_study2_data_save_dir = 'D:/IndexPen_data_Study1_Study2/user_study2_data/'
# participant_dir = 'participant_3'
# session_dir = 'session_5'

# full_session_dir_path = os.path.join(user_study2_data_dir, participant_dir, session_dir)

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

# trails_name = os.listdir(full_session_dir_path)
#
# #
# trail_path = os.path.join(full_session_dir_path, )

trail_path = "C:/Recordings/paper_test_trails/11_11_2021_13_58_53-Exp_HelloWorld-Sbj_someone-Ssn_0.dats"


indexpen_complete_raw = load_idp_raw(
    data_file_path=trail_path,
    DataStreamName=DataStreamName,
    reshape_dict=reshape_dict,
    exp_info_dict_json_path=exp_info_dict_json_path,
    rd_cr_ratio=0.8,
    ra_cr_ratio=0.8,
    all_categories=None,
    session_only=False
)

with open('../../../IndexPen_Training/analysis/Raw_Prediction/idp_raw', 'wb') as f:
    pickle.dump(indexpen_complete_raw, f)