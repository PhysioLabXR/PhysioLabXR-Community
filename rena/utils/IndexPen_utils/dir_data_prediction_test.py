import pickle

from sklearn.metrics import confusion_matrix

from preprocessing_utils import load_idp
import tensorflow as tf
import numpy as np

DataStreamName = 'TImmWave_6843AOP'
# data_dir_path = 'C:/Recordings/John_F-J'
data_dir_path = 'C:\Users\Haowe\OneDrive\Desktop\IndexPen_User_Study_Data\IndexPen_6000_Samples'
# data_dir_path = 'C:/Users/Haowe/OneDrive/Desktop/Day5_withNois'

# data_dir_path = 'C:/Recordings/John_A-J_test/2'

exp_info_dict_json_path = 'C:/Users/Haowe/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'

# save_data_dir = 'C:/Recordings/John_A-J_test/2/C-G_test'

reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
fs = 30
duration = 4
sample_num = fs * duration
# categories = [1, 2, 3, 4, 5]
categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_dict, Y, encoder = load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path,
                              sample_num, rd_cr_ratio=0.8, ra_cr_ratio=0.8, all_categories=None)
# prediction
model_path = '../../resource/mmWave/indexPen_model/3_7-20_Complex_Model_all_data_reg_all_large_kernal_size.h5'

model = tf.keras.models.load_model(model_path)

X_mmw_rD_test = X_dict[0]
X_mmw_rA_test = X_dict[1]

Y_pred1 = model.predict([X_mmw_rD_test, X_mmw_rA_test])
Y_pred = np.argmax(Y_pred1, axis=1)
Y_test = np.argmax(Y, axis=1)

cm = confusion_matrix(Y_test, Y_pred)