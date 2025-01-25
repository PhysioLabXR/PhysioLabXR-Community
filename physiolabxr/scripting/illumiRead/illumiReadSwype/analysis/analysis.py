import json
import os
import pickle
import warnings
from collections import defaultdict

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from physiolabxr.configs.config import default_group_name
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.user_utils import stream_in


def find_closes_time_index(target_timestamps, source_timestamp, return_diff=False):
    index_in_target = np.argmin(np.abs(target_timestamps - source_timestamp))
    if return_diff:
        return index_in_target, abs(target_timestamps[index_in_target] - source_timestamp)
    return index_in_target

TRIAL_TIMESTAMP_STD_TOLERANCE = 0.5
# user parameters ##############################################################
data_root = '/Users/apocalyvec/Library/CloudStorage/GoogleDrive-zl2990@columbia.edu/My Drive/Sweyepe/Data'
eyetracking_channel_names = json.load(open('/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/_presets/LSLPresets/VarjoEyeDataComplete.json', 'r'))['ChannelNames']

session_names = ['One', 'Two', 'Three', 'Four', 'Five']
condition_names = ['HandTap', 'GazePinch', 'Sweyepe']
participant_notes = {
    'P001': ''
}

# because in some of the earlier experiments, the absoluteTime in action info is not
# LSL's local clock, the log_time as one of the eyetracking channels is always used as the reference clock to
# sync the action info and the eyetracking stream

# start of the script #########################################################
participant_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if d.startswith('P')]
# sort by participant id
participant_dirs.sort(key=lambda x: int(os.path.basename(x)[1:]))

trial_data = pd.DataFrame(columns=['participant', 'session', 'condition', 'duration', 'wpm', 'finalUserInputEditDistance2Target', 'finalUserInputEditDistance2TargetNormalized'])

# first iterate over participants, then sessions, then trials
# word per minute of the first session for the three conditions
for p_i, p_dir in enumerate(participant_dirs):
    participant_id = os.path.basename(p_dir)
    session_dir_paths = [os.path.join(p_dir, s) for s in os.listdir(p_dir) if s in session_names]
    for s_i, s_dir in enumerate(session_dir_paths):
        print(f"Processing participant ({participant_id = }) {p_i + 1}/{len(participant_dirs)}, session {s_i + 1}/{len(session_dir_paths)}")
        sesion_number = session_names.index(os.path.basename(s_dir)) + 1
        # some times a session has multiple sub-sessions, if it is interupted
        subsession_dir_paths = [os.path.join(s_dir, x) for x in os.listdir(s_dir) if 'EasyExperiment' in x and '.' not in x]
        # sort the subsessions by the file name, because they are usually named <experiment name>1, <experiment name>2, ...
        subsession_dir_paths = sorted(subsession_dir_paths, key=lambda x: int(x.split('EasyExperiment')[-1]))

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
        # some times there are multiple physiolabxr recordings per session,
        stream_files = [[y for y in os.listdir(x) if y.endswith('.dats') or y.endswith('.p')] for x in subsession_dir_paths]
        stream_data_ = []
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
                    stream_data_.append(pickle.load(f))
            else:  # Load .dats file
                stream_obj = RNStream(dats_file_path)
                stream_data_.append(stream_obj.stream_in(jitter_removal=False))
                # Save .p file so that the next time we can load it directly
                with open(p_file_path, 'wb') as f:
                    pickle.dump(stream_data_[-1], f)
                    print(f"Saved stream pickle file to {p_file_path}")
        # concatenate the stream data
        stream_keys = stream_data_[0].keys()
        stream_data = defaultdict(lambda: [None, None])
        for key in stream_keys:
            stream_data[key][0] = np.concatenate([x[key][0] for x in stream_data_], axis=-1)  # 0 for the data, -1 is the time axis
            stream_data[key][1] = np.concatenate([x[key][1] for x in stream_data_])  # 0 for the time
        eyetracking_stream_status = stream_data['VarjoEyeTrackingLSL'][0][9]
        eyetracking_stream_x = stream_data['VarjoEyeTrackingLSL'][0][10]
        eyetracking_stream_timestamps = stream_data['VarjoEyeTrackingLSL'][0][eyetracking_channel_names.index('log_time')]

        # find the start of the first input in the trials
        # * Sweyepe: the first rwo when xKeyHitLocal is not -inf
        # also need to know if a trial is skipped
        # chunk the df by the trialIndex first
        trial_dfs = []
        for trial_index, trial_df in action_info_df.groupby('trialIndex'):
            # if EndState is in the column conditionType, then this trial is skipped
            if 'EndState' in trial_df['conditionType'].values:
                print(f'Find a skipped trial in participant {participant_id} session {sesion_number} trial {trial_index}')
                continue
            new_trial_row = {key: None for key in trial_data.columns}

            trial_df = trial_df[trial_df['conditionType'].isin(condition_names)]  # only keep the rows where conditionType is one of the condition_names

            # check the continuity of the trialTime and absoluteTime

            # trial validity checks ############################################
            assert np.max(np.diff(np.array(trial_df['trialTime'])) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"
            assert np.max(np.diff(np.array(trial_df['absoluteTime'])) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"

            # only keep the rows where conditionType is one of the condition_names
            user_inputs_step = trial_df['currentText'].drop_duplicates(keep='first')
            target_sentence = trial_df['targetText'].drop_duplicates(keep='first').iloc[0]
            if trial_df[trial_df['xKeyHitLocal'] != -np.inf].shape[0] == 0:
                warnings.warn(f"trial {trial_index} for Participant {participant_id} session {sesion_number} has no input. \n"
                              f"Deduplicated user input text are \n"
                              f"{trial_df['currentText'].drop_duplicates(keep='first')}. \n"
                              f"Drop this trial.")
                continue
            # TODO if the levenshtein distance between the target sentence and the last user input is too great, drop the trial
            new_trial_row['FinalUserInputEditDistance2Target'] = nltk.edit_distance(user_inputs_step.iloc[-1], target_sentence)
            new_trial_row['FinalUserInputEditDistance2TargetNormalized'] = new_trial_row['FinalUserInputEditDistance2Target'] / max(len(user_inputs_step.iloc[-1]), len(target_sentence))

            # get the trial condition
            new_trial_row['condition'] = trial_df['conditionType'].iloc[0]

            # find the first input
            if new_trial_row['condition'] == 'Sweyepe':
                first_input_index = trial_df[trial_df['xKeyHitLocal'] != -np.inf].index[0]  # this index is that of action_info_df's instead of trial_df's
            elif new_trial_row['condition'] == 'HandTap' or new_trial_row['condition'] == 'GazePinch':
                first_input_index = trial_df[trial_df['keyboardValue'] != np.nan].index[0]  # this index is that of action_info_df's instead of trial_df's
            else:
                raise Exception(f"Unknown condition {new_trial_row['condition']}")
            # find the last index where the currentText is changed, stripped of <color=green>, </color>
            trial_df.loc[:, 'currentText'] = trial_df['currentText'].str.replace('<color=green>', '').str.replace('</color>', '')  # Use .loc to avoid SettingWithCopyWarning
            # find indices of the last unique text
            last_unique_text_index = user_inputs_step.index[-1]  # this index is that of action_info_df's instead of trial_df's

            if last_unique_text_index <= first_input_index:
                raise Exception(f"Participant {participant_id} session {sesion_number} trial {trial_index} has no input. \n"
                                f"Deduplicated user input text are \n"
                                f"{trial_df['currentText'].drop_duplicates(keep='first')}.")
            trial_w_input_df = trial_df.loc[first_input_index:last_unique_text_index]

            # remove the rows of calibration
            # there are two ways to know if is calibrating,
            # * one is using the column 'eyeTrackingStatus'
            if 'eyeTrackingStatus' in trial_w_input_df.columns:
                trial_w_input_df = trial_w_input_df[trial_w_input_df['eyeTrackingStatus'] != 'calibrating']
            else:
                trial_start_stream_index, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_w_input_df['absoluteTime'].iloc[0], return_diff=True)
                assert diff < 5e-3, f"Participant {participant_id} session {sesion_number}: ActionInfo and EyetrackingStream might be out of sync"
                trial_end_stream_index, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_w_input_df['absoluteTime'].iloc[-1], return_diff=True)
                assert diff < 5e-3, f"Participant {participant_id} session {sesion_number}: ActionInfo and EyetrackingStream might be out of sync"
                eye_status_trial = eyetracking_stream_status[trial_start_stream_index:trial_end_stream_index]
                eye_x_trial = eyetracking_stream_x[trial_start_stream_index:trial_end_stream_index]
                eye_timestamps_trial = eyetracking_stream_timestamps[trial_start_stream_index:trial_end_stream_index]

                change_indices = np.where(np.diff(eye_status_trial) != 0)[0] + 1
                change_indices = np.concatenate(([0], change_indices, [len(eye_status_trial)]))
                durations = np.diff(eye_timestamps_trial[np.clip(change_indices, 0, len(eye_timestamps_trial) - 1)])
                total_invalid_duration = np.sum(durations[eye_status_trial[change_indices[:-1]] == 0])
            # TODO record per-word statistics

            # action_info_df.iloc[last_unique_text_index]['trialTime'] - action_info_df.iloc[first_input_index]['trialTime']

            # report the number of valid trials for each conditions that this session has

            new_trial_row['participant'] = participant_id
            new_trial_row['session'] = sesion_number
            new_trial_row['duration'] = trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0] - total_invalid_duration  # trial_df at this point is already cleaned

            trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
            trial_dfs.append(trial_df)
