import os
import numpy as np
from utils.IndexPen_utils.preprocessing_utils import load_idp
import pickle

data_save_dir = ''

all_data_dir_path = 'C:/Users/Haowe/OneDrive/Desktop/IndexPen_User_Study_Data/IndexPen_6000_Samples_complete/IndexPen_6000_Samples/'
exp_info_dict_json_path = '../../utils/IndexPen_utils/IndexPenExp.json'
reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
DataStreamName = 'TImmWave_6843AOP'


fs = 30
duration = 4
sample_num = fs * duration

subjects_data_dict = {}
subjects_label_dict = {}
subjects_group_dict = {}

subject_dirs = os.listdir(all_data_dir_path)
for subject_index, subject_dir in enumerate(subject_dirs):
    subject_X_dict = {}
    subject_Y = None

    subject_dir_path = os.path.join(all_data_dir_path, subject_dir)
    for root, subdirs, files in os.walk(subject_dir_path):
        # if there is a file in dir
        if len(files) != 0:
            # loop all data in the root
            X_dict, Y, encoder = load_idp(root, DataStreamName, reshape_dict, exp_info_dict_json_path,
                                          sample_num, rd_cr_ratio=None, ra_cr_ratio=None, all_categories=None)

            for channel in X_dict:
                if channel in subject_X_dict:
                    subject_X_dict[channel] = np.concatenate(
                        [subject_X_dict[channel], X_dict[channel]]
                        )
                else:
                    subject_X_dict[channel] = X_dict[channel]

            if subject_Y is not None:
                subject_Y = np.concatenate([subject_Y, Y])
            else:
                subject_Y = Y
            print('number of sample in dir: ', len(Y))

    # subject done

    subjects_data_dict[subject_dir] = subject_X_dict
    subjects_label_dict[subject_dir] = subject_Y
    subjects_group_dict[subject_dir] = np.array([subject_index]*len(subject_Y))



# all data extraction done

with open('8-13_5User_cr_(None,None)', 'wb') as f:
    pickle.dump([subjects_data_dict, subjects_label_dict, subjects_group_dict, encoder], f, protocol=4)



# all_data_dir_path = 'C:/Recordings/transfer_learning_test/transfer_learning_test_rnstream'
# for root, subdirs, files in os.walk(all_data_dir_path):
#     print(root)
#     print(subdirs)
#     print(files)
 