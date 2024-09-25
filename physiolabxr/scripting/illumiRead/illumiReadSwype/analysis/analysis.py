import os
import pickle

import pandas as pd
import numpy as np

from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.user_utils import stream_in

# user parameters ##############################################################
data_root = '/Users/apocalyvec/Library/CloudStorage/GoogleDrive-zl2990@columbia.edu/My Drive/Sweyepe/Data'
session_names = ['One', 'Two', 'Three', 'Four', 'Five']
condition_names = ['HandTap', 'GazePinch', 'Sweyepe']
participant_notes = {
    'P001': ''
}

# start of the script #########################################################
participant_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if d.startswith('P')]
# sort by participant id
participant_dirs.sort(key=lambda x: int(os.path.basename(x)[1:]))

trial_data = pd.DataFrame(columns=['participant', 'session', 'condition', 'duration', 'wpm'])
# word per minute of the first session for the three conditions
for p_i, p_dir in enumerate(participant_dirs):
    participant_id = os.path.basename(p_dir)
    print(f"Processing participant ({participant_id = }) {p_i + 1}/{len(participant_dirs)}")
    session_dir_paths = [os.path.join(p_dir, s) for s in os.listdir(p_dir) if s in session_names]
    for s_i, s_dir in enumerate(session_dir_paths):
        sesion_number = session_names.index(os.path.basename(s_dir)) + 1
        # some times a session has multiple sub-sessions, if it is interupted
        subsession_dir_paths = [os.path.join(s_dir, x) for x in os.listdir(s_dir) if 'EasyExperiment' in x and '.' not in x]


        # load the action info ###############################################
        action_info_files = [os.path.join(x, 'ActionInfo.csv') for x in subsession_dir_paths]
        action_info_dfs = [pd.read_csv(x) for x in action_info_files]
        # sort the action info files by the first absoluteTime
        action_info_dfs = sorted(action_info_dfs, key=lambda x: x['absoluteTime'].iloc[0])

        # if there are multiple subsessions, the trialIndex after the first subsession should be adjusted
        # to avoid the overlap of the trialIndex
        for i in range(1, len(action_info_dfs)):
            action_info_dfs[i]['trialIndex'] += action_info_dfs[i - 1]['trialIndex'].iloc[-1] + 1

        # load and concatenate the action info files
        action_info_df = pd.concat(action_info_dfs)

        # load the .dats ######################################################
        stream_files = [[y for y in os.listdir(x) if y.endswith('.dats') or y.endswith('.p')] for x in subsession_dir_paths]
        stream_data = []
        for subsession_d, dats_f in zip(subsession_dir_paths, stream_files):
            assert len(np.unique([x.split('.')[0] for x in dats_f])) == 1, f"Only one .dats or .p file with the same name (regardless of file extension) should be in, but got {stream_files} in {subsession_d}"
            stream_file_name = np.unique([x.split('.')[0] for x in dats_f])[0]  # file name without extension
            # load .p if it exists, otherwise load .dats (takes longer), and append to the stream_data,
            # if loaded from dats. save a .p for later

            # Check if .p file exists
            p_file_path = os.path.join(subsession_d, f"{stream_file_name}.p")
            dats_file_path = os.path.join(subsession_d, f"{stream_file_name}.dats")

            if os.path.exists(p_file_path):  # Load .p file
                with open(p_file_path, 'rb') as f:
                    stream_data.append(pickle.load(f))
            else:  # Load .dats file
                stream_obj = RNStream(dats_file_path)
                stream_data.append(stream_obj.stream_in(jitter_removal=False))
                # Save .p file for later
                with open(p_file_path, 'wb') as f:
                    pickle.dump(stream_data[-1], f)
                    print(f"Saved stream pickle file to {p_file_path}")

        # find the start of the first input in the trials
        # * Sweyepe: the first rwo when xKeyHitLocal is not -inf
        # also need to know if a trial is skipped
        # chunk the df by the trialIndex first
        trial_dfs = []
        for trial_index, trial_df in action_info_df.groupby('trialIndex'):
            # check the continuity of the trialTime and absoluteTime
            # only keep the rows where conditionType is one of the condition_names
            trial_df = trial_df[trial_df['conditionType'].isin(condition_names)]

            assert np.max(np.diff(np.array(trial_df['trialTime'])) < 0.5), f"The standard deviation of the timestamps is larger than {std_tolerance}"
            assert np.max(np.diff(np.array(trial_df['absoluteTime'])) < 0.5), f"The standard deviation of the timestamps is larger than {std_tolerance}"

            # get the trial condition
            condition = trial_df['conditionType'].iloc[0]

            # find the first input
            if condition == 'Sweyepe':
                first_input_index = trial_df[trial_df['xKeyHitLocal'] != -np.inf].index[0]
            elif condition == 'HandTap' or condition == 'GazePinch':
                first_input_index = trial_df[trial_df['keyboardValue'] != np.nan].index[0]

            # find the last index where the currentText is changed, stripped of <color=green>, </color>
            trial_df.loc[:, 'currentText'] = trial_df['currentText'].str.replace('<color=green>', '').str.replace('</color>', '')  # Use .loc to avoid SettingWithCopyWarning
            # find indices of the last unique text
            last_unique_text_index = trial_df['currentText'].drop_duplicates(keep='first').index[-1]

            # remove the rows of calibration
            # there are two ways to know if is calibrating, one is using the column 'eyeTrackingStatus'
            trial_df = trial_df.loc[first_input_index:]
            trial_dfs.append(trial_df)

