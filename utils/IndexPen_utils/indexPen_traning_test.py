from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from indexPen_model import make_model_dl_final, make_model_leo
from preprocessing_utils import load_idp
import numpy as np


import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.python.keras.models import load_model
from IPython.display import clear_output
from tensorflow.python.keras import Sequential, Model, Input
from tensorflow.python.keras.layers import TimeDistributed, Conv2D, BatchNormalization, MaxPooling2D, Flatten, \
    concatenate, LSTM, Dropout, Dense
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import os


DataStreamName = 'TImmWave_6843AOP'
data_dir_path = 'C:\Recordings\John_error_test_A-E'
exp_info_dict_json_path = 'C:/Users/HaowenWeiJohn/PycharmProjects/RealityNavigation/utils/IndexPen_utils/IndexPenExp.json'



reshape_dict = {
    'TImmWave_6843AOP': [(8, 16, 1), (8, 64, 1)]
}
fs = 30
duration = 4
sample_num = fs * duration

X_dict, Y, encoder = load_idp(data_dir_path, DataStreamName, reshape_dict, exp_info_dict_json_path, sample_num)


X_mmw_rD = X_dict[0]
X_mmw_rA = X_dict[1]

X_mmw_rD = (X_mmw_rD - np.min(X_mmw_rD)) / (np.max(X_mmw_rD) - np.min(X_mmw_rD))
X_mmw_rA = (X_mmw_rA - np.min(X_mmw_rA)) / (np.max(X_mmw_rA) - np.min(X_mmw_rA))

X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)
X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)


model = make_model_leo(classes=[1, 2, 3, 4, 5], points_per_sample=120)


es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint(
    filepath='AutoSave/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                                  '_') + '.h5',
    monitor='val_acc', mode='max', verbose=1, save_best_only=True)

history = model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                    validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                    epochs=200,
                    batch_size=16, callbacks=[es, mc], verbose=2, shuffle=True)

with open('log_hist.txt', 'w') as f:
    f.write(str(history.history))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
