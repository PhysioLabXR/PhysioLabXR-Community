import json

import matplotlib.pyplot as plt
from multiprocessing import freeze_support

from utils.data_utils import process_data

files = [
    # 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_16_43_45-Exp_realitynavigation-Sbj_0-Ssn_0 CLEANED.dats',
    # 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_16_52_54-Exp_realitynavigation-Sbj_0-Ssn_1 CLEANED.dats',
        'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_17_03_52-Exp_realitynavigation-Sbj_0-Ssn_2 CLEANED.dats',
         'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/03_22_2021_17_13_28-Exp_realitynavigation-Sbj_0-Ssn_3 CLEANED.dats',
         ]
EM_stream_name = 'Unity.RotationWheel.EventMarkers'
EEG_stream_name = 'BioSemi'
pre_stimulus_time = -.5
post_stimulus_time = 1.

EEG_stream_preset = json.load(open('D:\PycharmProjects\RealityNavigation\LSLPresets\BioSemiActiveTwo.json'))
notch_f0 = 60.
notch_band_demoninator = 200.
EEG_fresample = 50

if __name__ == '__main__':  # for windows all mp must be guarded by the main condition
    freeze_support()
    target_label = [2, 3, 4, 5, 6]
    evoked = process_data(files, EM_stream_name, EEG_stream_name, target_label, pre_stimulus_time, post_stimulus_time,
                     EEG_stream_preset, notes='Interleave')

    # title = 'EEG Original reference'
    # evoked.average().plot(titles=dict(eeg=title), time_unit='s')



    # plt.plot(epoched_EEG_timevector, epoched_EEG_average_trial_chan, c='b')
    #
    # plt.plot(epoched_EEG_timevector, epoched_EEG_average_trial_chan, c='r')

    # plt.fill_between(epoched_EEG_timevector, epoched_EEG_min_trial_chan, epoched_EEG_max_trial_chan, alpha=0.5)
    # plt.xlabel('Time (sec)')
    # plt.ylabel('Amplitude (\u03BCV)')
    # plt.title('All Session (baselined)')
    # plt.show()