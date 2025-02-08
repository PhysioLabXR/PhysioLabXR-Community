import json
import os
import pickle
import warnings
from collections import defaultdict
import re

import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from physiolabxr.configs.config import default_group_name
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.buffers import flatten
from physiolabxr.utils.user_utils import stream_in
import seaborn as sns

def find_closes_time_index(target_timestamps, source_timestamp, return_diff=False):
    index_in_target = np.argmin(np.abs(target_timestamps - source_timestamp))
    if return_diff:
        return index_in_target, target_timestamps[index_in_target] - source_timestamp
    return index_in_target

def check_if_is_known_problematic_trial(participant_id, session, trial_index):
    for problematic_trial in known_problematic_trial:
        if problematic_trial['participant'] == participant_id and problematic_trial['session'] == session and problematic_trial['trial_index'] == trial_index:
            return True
    return False


def strip_html_regex(html_text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', html_text)


def count_words(text):
    words = re.findall(r'\b\w+\b', text)
    return len(words)

def compute_invalid_duration(eye_status, eye_timestamps):
    change_indices = np.where(np.diff(eye_status) != 0)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [len(eye_status)]))
    durations = np.diff(eye_timestamps[np.clip(change_indices, 0, len(eye_timestamps) - 1)])
    invalid_duration = np.sum(durations[eye_status[change_indices[:-1]] == 0])
    return invalid_duration


TRIAL_TIMESTAMP_STD_TOLERANCE = 0.5
# user parameters ##############################################################
data_root = '/Users/apocalyvec/Library/CloudStorage/GoogleDrive-zl2990@columbia.edu/My Drive/Sweyepe/Data'
eyetracking_channel_names = json.load(open('/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/_presets/LSLPresets/VarjoEyeDataComplete.json', 'r'))['ChannelNames']
trial_data_export_path = os.path.join(data_root, 'trial_data.csv')

session_names = ['One', 'Two', 'Three', 'Four', 'Five']
condition_names = ['HandTap', 'GazePinch', 'Sweyepe']
participant_notes = {
    'P001': ''
}

known_problematic_trial = ([{'participant': 'P001', 'session': 1, 'trial_index': i} for i in range(12, 36)] +  # these are skipped trials
                           [
                            {'participant': 'P004', 'session': 4, 'trial_index': 0},  # the action info for this entire session is missing
                            {'participant': 'P013', 'session': 3, 'trial_index': 12},  # P013 has HandTap sessions recorded separately. Trial 12 is the last trial in the separate HandTap session. No actual trial is performed here.
                            {'participant': 'P013', 'session': 3, 'trial_index': 13},  # P013 actually only recorded for HandTap for session 3. EasyExperiment1 (whose first trial 13) is almost empty with no actual trials.
                            {'participant': 'P021', 'session': 1, 'trial_index': 0},  # P21 session 1's action info is almost empty. There's no trial data there
                            ])


# because in some of the earlier experiments, the absoluteTime in action info is not
# LSL's local clock, the log_time as one of the eyetracking channels is always used as the reference clock to
# sync the action info and the eyetracking stream

# start of the script #########################################################
participant_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if d.startswith('P')]
# sort by participant id
participant_dirs.sort(key=lambda x: int(os.path.basename(x)[1:]))

trial_data = pd.DataFrame(columns=['participant', 'session', 'trial_index', 'isValid', 'condition', 'duration',
                                   'wpm', 'finalUserInputEditDistance2Target', 'finalUserInputEditDistance2TargetNormalized',
                                   'targetSentence', 'userSentence',
                                   'overallCPM'] + [f'CPMWordLen={i}' for i in range(1, 30)] \
                                   + ['backspaceUsage', 'firstCandidateMissRate', 'allCandidateMissRate',])

trial_first_index_is_zero_counter = 0

# first iterate over participants, then sessions, then trials
# word per minute of the first session for the three conditions
trial_dfs = []  # breakpoint here to stop for each session
for p_i, p_dir in enumerate(participant_dirs):
    participant_id = os.path.basename(p_dir)
    session_dir_paths = [os.path.join(p_dir, s) for s in session_names if s in os.listdir(p_dir)]
    for s_i, s_dir in enumerate(session_dir_paths):
        print(f"Processing participant ({participant_id = }) {p_i + 1}/{len(participant_dirs)}, session {s_i + 1}/{len(session_dir_paths)}")
        session_number = session_names.index(os.path.basename(s_dir)) + 1
        # some times a session has multiple sub-sessions, if it is interupted

        subsession_dir_paths = [os.path.join(s_dir, x) for x in os.listdir(s_dir) if 'EasyExperiment' in x and '.' not in x]
        # sort the subsessions by the file name, because they are always named <experiment name>1, <experiment name>2, ...
        if len(subsession_dir_paths) > 1:
            subsession_dir_paths = sorted(subsession_dir_paths, key=lambda x: int(x.split('EasyExperiment')[-1]))

        # load the action info ###############################################
        action_info_files = [os.path.join(x, 'ActionInfo.csv') for x in subsession_dir_paths]
        # sort the action info files by parent folder's name. e.g., EasyExperiment1, EasyExperiment2, ...
        action_info_dfs = [pd.read_csv(x, low_memory=False) for x in action_info_files]

        # if there are multiple subsessions, the trialIndex after the first subsession should be adjusted
        for i in range(1, len(action_info_dfs)):
            action_info_dfs[i]['trialIndex'] += action_info_dfs[i - 1]['trialIndex'].iloc[-1] + 1

        # load and concatenate the action info files
        action_info_df = pd.concat(action_info_dfs)

        # load the .dats ######################################################
        # some times there are multiple physiolabxr recordings per session
        stream_files = [[y for y in os.listdir(x) if y.endswith('.dats') or y.endswith('.p')] for x in subsession_dir_paths]
        stream_data_ = []
        if len(flatten(stream_files)) == 0 and 'eyeTrackingStatus' not in action_info_df.columns:
            warnings(f"CRITICAL WARNING: Participant {participant_id} session {session_number} has no stream data and no eyeTrackingStatus column in the action info. Skipping this session.")
            continue
        elif len(flatten(stream_files)) == 0 and 'eyeTrackingStatus' in action_info_df.columns:
            warnings.warn(f"Participant {participant_id} session {session_number} has no stream data. But has eyeTrackingStatus column in the action info.")
            pass
        elif len(flatten(stream_files)) > 0:
            for subsession_d, dats_f in zip(subsession_dir_paths, stream_files):
                # if multiple .p and .dats exits for the same file name (e.g, file.dats and file.p), only keep one of them in the list
                if len(dats_f) == 0:
                    warnings.warn(f"Participant {participant_id} session {session_number} has no stream data in a subsession: {subsession_d}. Skipping this subsession.")
                    continue
                dats_f_deduplicated = {x.split('.')[0] for x in dats_f}
                assert len(np.unique([x.split('.')[0] for x in dats_f_deduplicated])) == 1, f"Participant {participant_id} session {session_number}: Only one .dats or .p file with the same name (regardless of file extension) should be in, but got {stream_files} in {subsession_d}"

                stream_file_name = list(dats_f_deduplicated)[0]  # file name without extension
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
        for trial_index, trial_df in action_info_df.groupby('trialIndex'):
            # if EndState is in the column conditionType, then this trial is skipped
            if 'EndState' in trial_df['conditionType'].values:
                print(f'Find a skipped trial in participant {participant_id} session {session_number} trial {trial_index}')
                continue
            new_trial_row = {}

            trial_df_filtered_with_condition = trial_df[trial_df['conditionType'].isin(condition_names)]  # only keep the rows where conditionType is one of the condition_names

            new_trial_row['participant'] = participant_id
            new_trial_row['session'] = session_number
            new_trial_row['trial_index'] = trial_index
            new_trial_row['isValid'] = True

            if len(trial_df_filtered_with_condition) <= 1:
                warnings.warn(f"Participant {participant_id} session {session_number} trial {trial_index} has no input. \n")
                new_trial_row['isValid'] = 'Trial after filtered by condition is empty'
                trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
                if not check_if_is_known_problematic_trial(participant_id, session_number, trial_index):
                    # if this exception is hit, investigate why the timestamps are not aligned. If it is a known issue, add it to the known_problematic_trial so this trial is skipped
                    raise Exception(
                        f"Participant {participant_id} session {session_number} trial {trial_index} does not have any valid rows after filtering. And it's not in the known problematic trials.")
                else:
                    continue

            # check the continuity of the trialTime and absoluteTime
            assert np.max(np.diff(np.array(trial_df_filtered_with_condition['trialTime'])) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"
            assert np.max(np.diff(np.array(trial_df_filtered_with_condition['absoluteTime'])) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"

            # only keep the rows where conditionType is one of the condition_names
            user_inputs_step = trial_df_filtered_with_condition['currentText'].drop_duplicates(keep='first')
            target_sentence = trial_df_filtered_with_condition['targetText'].drop_duplicates(keep='first').iloc[0]
            new_trial_row['targetSentence'] = target_sentence

            # add the trial data to the trial_data
            new_trial_row['userSentence'] = strip_html_regex(user_inputs_step.iloc[-1]) if user_inputs_step.iloc[-1] is not np.nan else np.nan
            # sometimes the first few rows of a trial_df, when grouped by trial_index, have the previous condition
            new_trial_row['condition'] = trial_df_filtered_with_condition['conditionType'].mode().item()

            deduplicated_user_inputs = trial_df_filtered_with_condition['currentText'].drop_duplicates(keep='first')
            if len(deduplicated_user_inputs) == 1 and np.isnan(deduplicated_user_inputs.iloc[0]):
                warnings.warn(f"trial {trial_index} for Participant {participant_id} session {session_number} has no input. \n"
                              f"Deduplicated user input text are \n"
                              f"{deduplicated_user_inputs}. \n"
                              f"Setting this trial to invalid.")
                new_trial_row['isValid'] = 'No user input'
                trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
                continue

            # TODO if the levenshtein distance between the target sentence and the last user input is too great, drop the trial
            new_trial_row['FinalUserInputEditDistance2Target'] = nltk.edit_distance(user_inputs_step.iloc[-1], target_sentence)
            new_trial_row['FinalUserInputEditDistance2TargetNormalized'] = new_trial_row['FinalUserInputEditDistance2Target'] / max(len(user_inputs_step.iloc[-1]), len(target_sentence))

            # find the first input
            if new_trial_row['condition'] == 'Sweyepe':
                first_input_index = trial_df_filtered_with_condition[trial_df_filtered_with_condition['xKeyHitLocal'] != -np.inf].index[0]  # this index is that of action_info_df's instead of trial_df's

            elif new_trial_row['condition'] == 'HandTap' or new_trial_row['condition'] == 'GazePinch':
                first_input_index = 0
                if 'eyeTrackingStatus' not in trial_df_filtered_with_condition.columns:  # need to check if the first ActionInfo input index is within the bounds of the stream timestamps
                    _, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_df_filtered_with_condition['absoluteTime'].iloc[first_input_index], return_diff=True)
                    if abs(diff) >= 5e-3 :
                        first_input_index = trial_df_filtered_with_condition[np.logical_not(pd.isna(trial_df_filtered_with_condition['keyboardValue']))].index[0]  # this index is that of action_info_df's instead of trial_df's
                if first_input_index == 0: trial_first_index_is_zero_counter += 1
            # else:
            #     raise Exception(f"Unknown condition {new_trial_row['condition']}")

            # find the last index where the currentText is changed, stripped of <color=green>, </color>
            trial_df_filtered_with_condition.loc[:, 'currentText'] = trial_df_filtered_with_condition['currentText'].str.replace('<color=green>', '').str.replace('</color>', '')  # Use .loc to avoid SettingWithCopyWarning
            # find indices of the last unique text
            last_unique_text_index = user_inputs_step.index[-1]  # this index is that of action_info_df's instead of trial_df's

            if last_unique_text_index <= first_input_index:
                raise Exception(f"Participant {participant_id} session {session_number} trial {trial_index} has no input. \n"
                                f"Deduplicated user input text are \n"
                                f"{trial_df_filtered_with_condition['currentText'].drop_duplicates(keep='first')}.")
            trial_w_input_df = trial_df_filtered_with_condition.loc[first_input_index:last_unique_text_index]  # the trial rows from the first input to the last unique text

            if new_trial_row['condition'] == 'Sweyepe' or new_trial_row['condition'] == 'GazePinch':  # removing the calibration time
            # if new_trial_row['condition'] == 'GazePinch':  # removing the calibration time
                # remove the rows of calibration
                # there are two ways to know if is calibrating,
                # * one is using the column 'eyeTrackingStatus'
                if 'eyeTrackingStatus' in trial_w_input_df.columns:
                    duration = trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0]
                    if len(trial_w_input_df[trial_w_input_df['eyeTrackingStatus'] != 'Available']) > 0:
                        eye_status_trial = np.array([1 if x == 'Available' else 0 for x in trial_w_input_df['eyeTrackingStatus']])
                        eye_timestamps_trial = trial_w_input_df['absoluteTime'].values
                        invalid_duration = compute_invalid_duration(eye_status_trial, eye_timestamps_trial)
                    else:
                        invalid_duration = 0
                else:  # 0 is invalid
                    trial_start_stream_index, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_w_input_df['absoluteTime'].iloc[0], return_diff=True)
                    try:
                        assert abs(diff) < 5e-3
                        trial_end_stream_index, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_w_input_df['absoluteTime'].iloc[-1], return_diff=True)
                        assert abs(diff) < 5e-3
                    except AssertionError as e:
                        warnings.warn(f"Participant {participant_id} session {session_number}: ActionInfo and EyetrackingStream might be out of sync.")
                        new_trial_row['isValid'] = 'Timestamp mismatch between stream and action info'
                        trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
                        if not check_if_is_known_problematic_trial(participant_id, session_number, trial_index):
                            # if this exception is hit, investigate why the timestamps are not aligned. If it is a known issue, add it to the known_problematic_trial so this trial is skipped
                            raise Exception(f"Participant {participant_id} session {session_number} trial {trial_index} has a timestamp mismatch. And it's not in the known problematic trials.")
                        else:
                            continue

                    eye_status_trial = eyetracking_stream_status[trial_start_stream_index:trial_end_stream_index]
                    eye_timestamps_trial = eyetracking_stream_timestamps[trial_start_stream_index:trial_end_stream_index]

                    invalid_duration = compute_invalid_duration(eye_status_trial, eye_timestamps_trial)

                    # create an eye status array fitted to the absoluteTime in action info
                    stream_eye_status_df = pd.DataFrame({'timestamp': eye_timestamps_trial, 'status': eye_status_trial})
                    merged_df = pd.merge_asof(pd.DataFrame({'timestamp': trial_w_input_df['absoluteTime']}).sort_values('timestamp'),
                                              stream_eye_status_df.sort_values('timestamp'),
                                              on='timestamp',
                                              direction='nearest')
                    # add the fitted eye status to the trial_w_input_df
                    trial_w_input_df['eyeTrackingStatusFittedFromStream'] = merged_df['status']

            if new_trial_row['condition'] == 'Sweyepe':
                    eye_status_col_name = 'eyeTrackingStatus' if 'eyeTrackingStatus' in trial_w_input_df.columns else 'eyeTrackingStatusFittedFromStream'
                    assert eye_status_col_name in trial_w_input_df.columns, f"Participant {participant_id} session {session_number} trial {trial_index} does not have the column {eye_status_col_name}."

                    key_hit = trial_w_input_df['xKeyHitLocal'].values
                    candidate1_is_na = pd.isna(trial_w_input_df['candidate1']).values
                    eye_status = trial_w_input_df[eye_status_col_name].values

                    eye_hit_status = np.array([(0 if abs(x_key_hit) == np.inf and x_cndt_is_na else 1) for x_key_hit, x_cndt_is_na in zip(key_hit, candidate1_is_na)])
                    hit_invalid_duration = compute_invalid_duration(eye_hit_status, trial_w_input_df['absoluteTime'].values)

                    eye_hit_calib_status = np.array([(0 if ((abs(x_key_hit) == np.inf and x_cndt_is_na ) or x_status == 0 )else 1) for x_key_hit, x_cndt_is_na, x_status in zip(key_hit, candidate1_is_na, eye_status)])
                    hit_calib_invalid_duration = compute_invalid_duration(eye_hit_calib_status, trial_w_input_df['absoluteTime'].values)

                    # print(f"Total duration: {trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0]}. Hit ID: {hit_invalid_duration}. Hit calib ID: {hit_calib_invalid_duration}")
                    invalid_duration = hit_calib_invalid_duration
            else:
                invalid_duration = 0

            new_trial_row['duration'] = trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0] - invalid_duration  # trial_df at this point is already cleaned
            new_trial_row['wpm'] = count_words(new_trial_row['userSentence']) / new_trial_row['duration'] * 60

            # 1. TODO overall CPM, CPM as a function word length
            new_trial_row['overallCPM'] = len(new_trial_row['userSentence']) / new_trial_row['duration'] * 60

            # find all the words in the sentence
            words = re.findall(r'\b\w+\b', new_trial_row['userSentence'])
            trial_w_input_df.loc[:, 'currentWords']= trial_w_input_df['currentText'].apply(lambda x: re.findall(r'\b\w+\b', x))
            trial_w_input_df.loc[:, 'nWords'] = trial_w_input_df['currentWords'].apply(lambda x: len(x))

            trial_w_input_df_copy = trial_w_input_df.copy()
            for w in words:
                # find the index when w first appear in trial_w_input_df['currentText']
                try:
                    w_first_appear_index = trial_w_input_df[trial_w_input_df['currentWords'].apply(lambda x: w in x)].index[0]
                except IndexError as e:
                    pass
                if new_trial_row['condition'] != 'Sweyepe':
                    # find the last time nWords increment in the series up to w_first_appear_index, this is the beginning letter of the new word first being typed
                    nWords_last_increment_index = trial_w_input_df.loc[:w_first_appear_index, 'nWords'].idxmax()
                    assert w_first_appear_index >= nWords_last_increment_index, f"Participant {participant_id} session {session_number} trial {trial_index} word {w} first appear index is smaller than the last nWords increment index."
                    w_duration = trial_w_input_df['absoluteTime'][w_first_appear_index] - trial_w_input_df['absoluteTime'][nWords_last_increment_index]
                    new_trial_row[f'CPMWordLen={len(w)}'] = len(w) / w_duration * 60
                else:  # in Sweyepe, the start-typing time for a word is when the sweyepe trace starts
                    pass


            # 2. TODO Backspace usage

            # 3. TODO Sweyepe first candidate miss rate, all candidate miss rate

            # 4. TODO information transfer rate

            # 5. TODO

            # action_info_df.iloc[last_unique_text_index]['trialTime'] - action_info_df.iloc[first_input_index]['trialTime']

            # report the number of valid trials for each conditions that this session has

            trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
            trial_dfs.append(trial_df_filtered_with_condition)


# save the trial_data
trial_data.to_csv(trial_data_export_path, index=False)

# plot the wpm as a function of trial index
# sns
trial_data_wpm_na_dropped = trial_data.dropna(subset=['wpm'])
sns.boxplot(data=trial_data_wpm_na_dropped, x="session", y="wpm", hue="condition")
plt.ylim(0, 60)
plt.show()

sns.boxplot(data=trial_data_wpm_na_dropped, x="session", y="cpm", hue="condition")
plt.ylim(0, 60)
plt.show()