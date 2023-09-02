# import pickle
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
#
# from physiolabxr.scripting.Examples.P300SpellerDemo.Cyton8ChannelsConfig import eeg_channel_names
# from physiolabxr.scripting.Examples.P300SpellerDemo.P300Speller_params import *
# from physiolabxr.scripting.Examples.P300SpellerDemo.P300Speller_utils import p300_speller_process_raw_data, rebalance_classes, \
#     confusion_matrix, visualize_eeg_epochs
# import mne
#
# model = LogisticRegression()
#
# file_name = '02_15_2023_00_32_12_train_raw.pickle'
#
# with (open(file_name, "rb")) as openfile:
#      raw = pickle.load(openfile)
#
# raw_processed = p300_speller_process_raw_data(raw, l_freq=1, h_freq=50, notch_f=60, picks='eeg')
# flashing_events = mne.find_events(raw_processed, stim_channel='P300SpellerTargetNonTargetMarker')
#
# epoch = mne.Epochs(raw_processed, flashing_events, tmin=-0.1, tmax=1, baseline=(-0.1, 0), event_id=event_id,
#                    preload=True)
#
# visualize_eeg_epochs(epoch, event_id,event_color, eeg_channel_names)
#
# x = epoch.get_data(picks='eeg')
# y = epoch.events[:, 2]
# x, y = rebalance_classes(x, y, by_channel=True)
# x = x.reshape(x.shape[0], -1)
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size)
# model.fit(x_train, y_train)
# y_pred = model.predict(x_test)
# confusion_matrix(y_test, y_pred)
#
#
#
#
# colum_row_marker = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingRowOrColumMarker')[:,-1]
# colum_row_index = mne.find_events(raw_processed, stim_channel='P300SpellerFlashingRowColumIndexMarker')[:,-1]
#
# merged_list = [(colum_row_marker[i], colum_row_index[i]) for i in range(0, len(colum_row_index))]
#
#
#
# x = epoch.get_data(picks='eeg')
# x = x.reshape(x.shape[0], -1)
# y_pred = model.predict_proba(x)[:,1]
#
# row_1 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,1)]]
# row_2 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,2)]]
# row_3 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,3)]]
# row_4 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,4)]]
# row_5 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,5)]]
# row_6 = y_pred[[i for i, j in enumerate(merged_list) if j == (ROW_FLASHING_MARKER,6)]]
#
# col_1 = y_pred[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER,1)]]
# col_2 = y_pred[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER,2)]]
# col_3 = y_pred[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER,3)]]
# col_4 = y_pred[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER,4)]]
# col_5 = y_pred[[i for i, j in enumerate(merged_list) if j == (COL_FLASHING_MARKER,5)]]
#
# target_row = np.argmax([row_1.mean(), row_2.mean(), row_3.mean(), row_4.mean(), row_5.mean(), row_6.mean()])
# target_col = np.argmax([col_1.mean(), col_2.mean(), col_3.mean(), col_4.mean(), col_5.mean()])