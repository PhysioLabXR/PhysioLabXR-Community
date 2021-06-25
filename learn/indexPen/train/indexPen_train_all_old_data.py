import datetime

from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from tensorflow.keras.callbacks import CSVLogger
import indexPen_make_model


# load_data_dir = ['C:/Users/Haowe/OneDrive/Desktop/IndexPenData/ag_6100_corrupt_frame_removal',
#                  'C:/Users/Haowe/OneDrive/Desktop/IndexPenData/hw_6300_corrupt_frame_removal',
#                  'C:/Users/Haowe/OneDrive/Desktop/IndexPenData/zl_5800_corrupt_frame_removal']

load_data_dir = ['C:/Users/Haowe/OneDrive/Desktop/IndexPenData/ag_6100_corrupt_frame_removal',
                 'C:/Users/Haowe/OneDrive/Desktop/IndexPenData/hw_6300_corrupt_frame_removal',
                 'C:/Users/Haowe/OneDrive/Desktop/IndexPenData/zl_5800_corrupt_frame_removal']


X_mmw_rD = None
X_mmw_rA = None
Y = None
for data_path in load_data_dir:
  with open(data_path, 'rb') as f:
    X_dict_temp, Y_temp, encoder_temp = pickle.load(f)

  X_mmw_rD_temp = X_dict_temp['range_doppler']
  X_mmw_rA_temp = X_dict_temp['range_azi']

  if X_mmw_rD is None:
    X_mmw_rD = X_mmw_rD_temp
    X_mmw_rA = X_mmw_rA_temp
    Y = Y_temp
  else:
    X_mmw_rD = np.concatenate([X_mmw_rD, X_mmw_rD_temp])
    X_mmw_rA = np.concatenate([X_mmw_rA, X_mmw_rA_temp])
    Y = np.concatenate([Y, Y_temp])
  print('load file: ', data_path)

del X_dict_temp
del Y_temp
# del encoder_temp

# X_mmw_rD = X_dict[0]
# X_mmw_rA = X_dict[1]

# X_mmw_rD = X_dict['range_doppler']
# X_mmw_rA = X_dict['range_azi']


print(np.min(X_mmw_rD))
print(np.max(X_mmw_rD))

print(np.min(X_mmw_rA))
print(np.max(X_mmw_rA))


# rD_min = np.min(X_mmw_rD)
# rD_max = np.max(X_mmw_rD)

# rA_min = np.min(X_mmw_rA)
# rA_max = np.max(X_mmw_rA)


rD_min = -1000
rD_max = 1500

rA_min = 0
rA_max = 2500

X_mmw_rD = (X_mmw_rD - rD_min) / (rD_max - rD_min)
X_mmw_rA = (X_mmw_rA - rA_min) / (rA_max - rA_min)

X_mmw_rD_train, X_mmw_rD_test, Y_train, Y_test = train_test_split(X_mmw_rD, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)
del X_mmw_rD

X_mmw_rA_train, X_mmw_rA_test, Y_train, Y_test = train_test_split(X_mmw_rA, Y, test_size=0.20, random_state=3,
                                                                  shuffle=True)
del X_mmw_rA
del Y


sample_num = 120
rd_shape = (8, 16)
ra_shape = (8, 64)
# model = make_model_leo(classes=[1, 2, 3, 4, 5], points_per_sample=sample_num)

idp_complete_classes = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O',  # accuracy regression
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y',  # accuracy regression
    'Z', 'Spc', 'Bspc', 'Ent', 'Act'
]

model = indexPen_make_model.make_model_dl_final()

# model = make_model_dl_final()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
csv_logger = CSVLogger("model_history_log.csv", append=True)
mc = ModelCheckpoint(
    filepath='model/' + str(datetime.datetime.now()).replace(':', '-').replace(' ',
                                                                                  '_') + '.h5',
    monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

history = model.fit([X_mmw_rD_train, X_mmw_rA_train], Y_train,
                    validation_data=([X_mmw_rD_test, X_mmw_rA_test], Y_test),
                    epochs=20000,
                    batch_size=64, callbacks=[es, mc, csv_logger], verbose=2, shuffle=True)


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

Y_pred1 = model.predict([X_mmw_rD_test, X_mmw_rA_test])

Y_pred = np.argmax(Y_pred1, axis=1)
Y_test = np.argmax(Y_test, axis=1)
test_acc = accuracy_score(Y_test, Y_pred)
print(test_acc)
cm = confusion_matrix(Y_test, Y_pred)
