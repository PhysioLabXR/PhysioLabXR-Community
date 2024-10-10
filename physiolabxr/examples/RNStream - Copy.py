from physiolabxr.utils.RNStream import RNStream

test_rns = RNStream('C:/Users/Amadeus/Documents/Recordings/05_26_2023_16_13_38-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
test_reloaded_data = test_rns.stream_in_stepwise(None, {}, None, jitter_removal=False)

another_rns = RNStream('C:/Users/Amadeus/Documents/Recordings/05_26_2023_16_13_38-Exp_myexperiment-Sbj_someone-Ssn_0.dats')
another_reloaded_data = another_rns.stream_in_stepwise(None, {}, None)

result_rns = RNStream('C:/Users/Amadeus/Documents/Recordings/results.dats')
result_rns.stream_out(test_reloaded_data)
result_rns.stream_out(another_reloaded_data)

results_reloaded_data = result_rns.stream_in()