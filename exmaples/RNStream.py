from utils.data_utils import RNStream

test_rns = RNStream('C:/Recordings/05_27_2021_23_13_33-Exp_myexperiment-Sbj_video_test-Ssn_0.dats')
test_reloaded_data = test_rns.stream_in()




# another_rns = RNStream('C:/Recordings/03_22_2021_00_00_55-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
# another_reloaded_data = another_rns.stream_in()
#
# result_rns = RNStream('C:/Recordings/results.dats')
# result_rns.stream_out(test_reloaded_data)
# result_rns.stream_out(another_reloaded_data)
#
# results_reloaded_data = result_rns.stream_in()

