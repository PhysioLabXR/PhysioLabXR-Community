from rena.utils.data_utils import RNStream

test_rns = RNStream('C:/Recordings/10_09_2021_18_50_45-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
test_reloaded_data = test_rns.stream_in(jitter_removal=False)

another_rns = RNStream('C:/Recordings/03_22_2021_00_00_55-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
another_reloaded_data = another_rns.stream_in()

result_rns = RNStream('C:/Recordings/results.dats')
result_rns.stream_out(test_reloaded_data)
result_rns.stream_out(another_reloaded_data)

results_reloaded_data = result_rns.stream_in()