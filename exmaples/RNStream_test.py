import pickle
import time
from utils.data_utils import RNStream

# recording_buffer = pickle.load(open('C:/Recordings/03_21_2021_22_48_25-Exp_Unity.RealityNavigationHotel.EventMarkers-Sbj_someone-Ssn_0.p', 'rb'))

# rns.stream_out(recording_buffer)



rns = RNStream('C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_16_43_45-Exp_realitynavigation-Sbj_0-Ssn_0.dats')

start_time = time.time()
reloaded_buffer = rns.stream_in(ignore_stream=('monitor1', '0'))
print('reload with ignore took {0}'.format(time.time() - start_time))

start_time = time.time()
reloaded_buffer = rns.stream_in()
print('reload all took {0}'.format(time.time() - start_time))
