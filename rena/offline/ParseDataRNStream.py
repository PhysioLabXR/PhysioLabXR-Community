from rena.utils.data_utils import RNStream

test_rns = RNStream('/Users/Leo/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_17_03_52-Exp_realitynavigation-Sbj_0-Ssn_2.dats')
reloaded_data = test_rns.stream_in(ignore_stream=('monitor1', '0'))

