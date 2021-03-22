import pickle
import numpy as np

recording_buffer = pickle.load(open('C:/Recordings/03_21_2021_22_37_43-Exp_Unity.RealityNavigationHotel.EventMarkers-Sbj_someone-Ssn_0.p', 'rb'))

m = np.mean(recording_buffer['0'][1] - recording_buffer['0'][2]) * 10 **3
std = np.std(recording_buffer['0'][1] - recording_buffer['0'][2]) * 10 **3
