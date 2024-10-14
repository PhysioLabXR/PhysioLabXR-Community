import numpy as np

from physiolabxr.scripting.PupilTensorflowModel import PupilTensorflowModel

X = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL.npy')
y = np.load('C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/SingleTrials/epochs_pupil_raw_condition_RSVP_DL_labels.npy')

model = PupilTensorflowModel('D:\PycharmProjects\ReNaAnalysis\Learning\Model\Pupil_ANN')

buffer_data = np.concatenate(X[:5], axis=0)  # dummy buffer data

preprocess_data = model.preprocess(buffer_data)
y_pred = model.predict(preprocess_data)

# model.model.evaluate(x = X, y = y)
#
# [plt.plot(xx) for xx in x]
# plt.show()