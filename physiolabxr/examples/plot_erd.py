import pickle
import numpy as np
from physiolabxr.scripting.physio.epochs import visualize_erd, visualize_epochs

srate = 250
tmin = 2
reject = 400

epochs_data = np.load(r'E:\Data\MotorImageryParticipantDataDir\x\epochs_data.npy')
labels = np.load(r'E:\Data\MotorImageryParticipantDataDir\x\labels.npy')
labels = np.array(['Right' if l == 0 else 'Left' for l in labels])

picks = ['F3', 'Fz', 'F4', 'C3', 'Cz', 'C4', 'P3', 'P4']
tmax = tmin + epochs_data.shape[-1] / srate


# apply rejection
epoch_p2p = np.max(epochs_data, axis=-1) - np.min(epochs_data, axis=-1)
epoch_rejects = np.any(epoch_p2p > reject, axis=1)
epochs_data = epochs_data[~epoch_rejects]
labels = labels[~epoch_rejects]

epochs = {event: epochs_data[np.argwhere(labels == event)[:, 0]] for event in np.unique(labels)}

visualize_epochs(epochs, tmin=tmin, tmax=tmax, picks=picks)
visualize_erd(epochs, tmin, tmax, srate, picks=picks)
