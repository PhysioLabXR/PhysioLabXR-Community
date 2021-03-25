import pickle
import numpy as np

from utils.data_utils import RNStream

file_path = 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_16_52_54-Exp_realitynavigation-Sbj_0-Ssn_1.dats'

rns = RNStream(file_path)
data = rns.stream_in(ignore_stream=('monitor1', '0'))
