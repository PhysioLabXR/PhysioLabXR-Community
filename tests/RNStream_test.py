import pickle

from utils.data_utils import RNStream

recording_buffer = pickle.load(open('C:/Recordings/03_21_2021_22_48_25-Exp_Unity.RealityNavigationHotel.EventMarkers-Sbj_someone-Ssn_0.p', 'rb'))

rns = RNStream('C:/Recordings/stream_test.rn')

rns.stream_out(recording_buffer)


reloaded_buffer = rns.stream_in()