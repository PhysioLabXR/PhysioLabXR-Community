import argparse
import copy
import json
import os
import pickle
import warnings
from collections import defaultdict
import re

import editdistance
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.buffers import flatten
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

def reconstruct_input_stream(input_snapshots):
    """
    Reconstructs the input stream from a list of input box snapshots.

    :param input_snapshots: A list of strings representing the content of the input box at each key event.
    :return: A string representing the reconstructed input stream.
    """
    input_stream = ""
    prev_text = ""

    for current_text in input_snapshots:
        if current_text == prev_text:  # No change
            continue

        if len(current_text) > len(prev_text):  # Insertion detected
            # Find what was inserted (bulk insert or single char)
            inserted_text = current_text[len(prev_text):]
            input_stream += inserted_text  # Append all inserted characters

        elif len(current_text) < len(prev_text):  # Deletion detected
            input_stream += "←"  # Represent any delete (single char or bulk) as "←"

        prev_text = current_text  # Update previous text

    return input_stream  # No extra spaces, fully compatible with `compute_error_rates`


def compute_total_error_rates(presented_text, input_stream, transcribed_text):
    """
    Computes the total error rate based on the input stream and final transcribed text.

    :param presented_text: The target sentence the user was supposed to type.
    :param input_stream: The full sequence of typed characters, including corrections.
    :param transcribed_text: The final output after typing.
    :return: A dictionary with error rates.
    """
    # Step 1: Compute INF (Errors that remain in transcribed text)
    INF = editdistance.eval(presented_text, transcribed_text)

    # Step 2: Count all correctly typed characters
    C = len(transcribed_text) - INF

    # Step 3: Identify corrected errors (IF) and Fixes (F)
    IF, F = 0, 0
    backspace_count = input_stream.count("←")  # Assume backspace is represented as "←"

    for i in range(len(input_stream)):
        if input_stream[i] == "←":
            F += 1  # Each backspace is a fix
        elif i > 0 and input_stream[i-1] == "←":
            IF += 1  # Count the corrected characters

    # Step 4: Compute total error rate
    total_error_rate = ((IF + INF) / (C + INF + IF)) * 100
    corrected_error_rate = (IF / (C + INF + IF)) * 100
    not_corrected_error_rate = (INF / (C + INF + IF)) * 100

    return {
        "Total Error Rate (%)": total_error_rate,
        "Corrected Error Rate (%)": corrected_error_rate,
        "Not Corrected Error Rate (%)": not_corrected_error_rate,
        "Fixes (F)": F,
        "Incorrect Fixed (IF)": IF,
        "Incorrect Not Fixed (INF)": INF
    }


def compute_msd_error_rates(presented_text, transcribed_text):
    """
    Computes the MSD Error Rate (Old and New) for a given presented text and transcribed text.

    :param presented_text: The target sentence the user was supposed to type.
    :param transcribed_text: The final output after typing.
    :return: A dictionary with both old and new MSD error rates.
    """
    # Compute Minimum String Distance (MSD)
    MSD = editdistance.eval(presented_text, transcribed_text)

    # Compute New MSD Error Rate using Mean Alignment String Length
    mean_alignment_length = (len(presented_text) + len(transcribed_text)) / 2  # Approximation
    new_msd_error_rate = (MSD / mean_alignment_length) * 100 if mean_alignment_length > 0 else 0

    return new_msd_error_rate


parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--reload_data", action="store_true", help="Reload data from the disk.")
group.add_argument("--no_reload_data", action="store_false", dest="reload_data", help="Load from cache instead of reloading.")
parser.add_argument("--data_root", type=str, required=True, help="The root directory of the data. If not specified, use the default data root.")
parser.add_argument("--only_run_participants",  nargs='+', type=str, required=False, help="Only run the specified participants. Comma separated list of participant ids.", default=None)
parser.add_argument("--skip_participants",  nargs='+', type=str, required=False, help="Skipped the specified participants. Comma separated list of participant ids.", default=None)
args = parser.parse_args()

# check only run and sipped participants doesn't overlap

TRIAL_TIMESTAMP_STD_TOLERANCE = 0.5
# user parameters ##############################################################
# data_root = '/Users/apocalyvec/Library/CloudStorage/GoogleDrive-zl2990@columbia.edu/My Drive/Sweyepe/Data'
data_root = args.data_root
eyetracking_channel_names = json.load(open('/Users/apocalyvec/PycharmProjects/PhysioLabXR/physiolabxr/_presets/LSLPresets/VarjoEyeDataComplete.json', 'r'))['ChannelNames']
trial_data_export_path = os.path.join(data_root, 'trial_data.csv')
reload_data = args.reload_data

session_names = ['One', 'Two', 'Three', 'Four', 'Five']
swype_conditions = ['Sweyepe', 'PartialSwipe', 'HandSwipe']
eyetracking_conditions = ['Sweyepe', 'PartialSwipe', 'GazePinch']

# use the data root name to determine which study is this
# TODO add study 3 here
if 'Study2' in os.path.basename(data_root):
    study = 2
    condition_names = ['HandTap', 'HandSwipe', 'PartialSwipe']  # HandTap here is with word completion
else:
    study = 1
    condition_names = ['HandTap', 'GazePinch', 'Sweyepe']

known_problematic_trial = ([{'participant': 'P001', 'session': 1, 'trial_index': i} for i in range(12, 36)] +  # these are skipped trials
                           [
                            {'participant': 'P004', 'session': 4, 'trial_index': 0},  # the action info for this entire session is missing
                            {'participant': 'P013', 'session': 3, 'trial_index': 12},  # P013 has HandTap sessions recorded separately. Trial 12 is the last trial in the separate HandTap session. No actual trial is performed here.
                            {'participant': 'P013', 'session': 3, 'trial_index': 13},  # P013 actually only recorded for HandTap for session 3. EasyExperiment1 (whose first trial 13) is almost empty with no actual trials.
                            {'participant': 'P021', 'session': 1, 'trial_index': 0},  # P21 session 1's action info is almost empty. There's no trial data there
                            {'participant': 'P025', 'session': 2, 'trial_index': 61},  # this is an extra row confused with TechniqueIntroStates
                            {'participant': 'P025', 'session': 3, 'trial_index': 61},  # same as above
                            {'participant': 'P028', 'session': 1, 'trial_index': 24},  # same as above
                            {'participant': 'P028', 'session': 1, 'trial_index': 49},  # same as above
                            {'participant': 'P029', 'session': 1, 'trial_index': 12},  # same as above
                            {'participant': 'P032', 'session': 1, 'trial_index': 61},  # same as above

                            {'participant': 'P2_004', 'session': 1, 'trial_index': 24},  # same as above
                            {'participant': 'P2_008', 'session': 1, 'trial_index': 24},  # same as above
                            ])
only_run_participants = args.only_run_participants
skipped_participants = args.skip_participants

if only_run_participants is not None and skipped_participants is not None:
    assert all([x not in args.skip_participants for x in args.only_run_participants])


# because in some of the earlier experiments, the absoluteTime in action info is not
# LSL's local clock, the log_time as one of the eyetracking channels is always used as the reference clock to
# sync the action info and the eyetracking stream

# start of the script #########################################################
if __name__ == '__main__':
    if reload_data:
        participant_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if d.startswith('P')]
        if skipped_participants:
            participant_dirs = [d for d in participant_dirs if os.path.basename(d) not in skipped_participants]
        # sort by participant id
        participant_dirs.sort(key=lambda x: int(os.path.basename(x)[1:]))

        trial_data = pd.DataFrame(columns=['participant', 'session', 'trial_index', 'condition_trial_index', 'isValid', 'condition', 'duration',
                                           'wpm', 'finalUserInputEditDistance2Target', 'finalUserInputEditDistance2TargetNormalized',
                                           'targetSentence', 'userSentence',
                                           'overallCPM'] + [f'CPMWordLen={i}' for i in range(1, 30)] \
                                           + ['backspaceUsage', 'firstCandidateMissRate', 'allCandidateMissRate', 'sweyepe_first_candidate_match_rate', 'sweyepe_any_candidate_match_rate', 'sweyepe_all_candidate_miss_rate',
                                              'numDeletePress', 'numDeletePressPerChar'])

        trial_first_index_is_zero_counter = 0

        # first iterate over participants, then sessions, then trials
        # word per minute of the first session for the three conditions
        trial_dfs = []  # breakpoint here to stop for each session
        for p_i, p_dir in enumerate(participant_dirs):
            participant_id = os.path.basename(p_dir)
            session_dir_paths = [os.path.join(p_dir, s) for s in session_names if s in os.listdir(p_dir)]
            if only_run_participants is not None and participant_id not in only_run_participants:
                print(f"Skipping participant {participant_id} because it's not in the only_run_participants list.")
                continue
            for s_i, s_dir in enumerate(session_dir_paths):
                print(f"Processing participant ({participant_id = }) {p_i + 1}/{len(participant_dirs)}, session {s_i + 1}/{len(session_dir_paths)}")
                session_number = session_names.index(os.path.basename(s_dir)) + 1

                # sometimes a session has multiple sub-sessions, if it is interupted

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
                n_sweyepe_words = 0
                n_sweyepe_first_candidate_matched = 0
                n_sweyepe_any_candidates_matched = 0
                n_sweyepe_all_candidate_missed = 0
                condition_trial_index = defaultdict(int)  # keeps track of the index for each trial for a given condition
                for trial_index, trial_df in action_info_df.groupby('trialIndex'):
                    # if EndState is in the column conditionType, then this trial is skipped
                    if 'EndState' in trial_df['conditionType'].values:
                        print(f'Find a skipped trial in participant {participant_id} session {session_number} trial {trial_index}')
                        continue
                    new_trial_row = {}

                    # TODO designate the columns trialTime and absoluteTime to be float columns, this will fix the P2_008

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
                                f"Participant {participant_id} session {session_number} trial {trial_index} does not have any valid rows after filtering. And it's not in the known problematic trials."
                                f"Row content is {trial_df_filtered_with_condition}.")
                        else:
                            continue

                    # check the continuity of the trialTime and absoluteTime
                    try:
                        assert np.max(np.diff(np.array(trial_df_filtered_with_condition['trialTime'], dtype=float)) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"
                        assert np.max(np.diff(np.array(trial_df_filtered_with_condition['absoluteTime'], dtype=float)) < TRIAL_TIMESTAMP_STD_TOLERANCE), f"The standard deviation of the timestamps is larger than {TRIAL_TIMESTAMP_STD_TOLERANCE}"
                    except TypeError as e:
                        raise e
                    # only keep the rows where conditionType is one of the condition_names
                    user_inputs_step = trial_df_filtered_with_condition['currentText'].drop_duplicates(keep='first')
                    target_sentence = trial_df_filtered_with_condition['targetText'].drop_duplicates(keep='first').iloc[0]
                    new_trial_row['targetSentence'] = target_sentence

                    # add the trial data to the trial_data
                    new_trial_row['userSentence'] = strip_html_regex(user_inputs_step.iloc[-1]) if user_inputs_step.iloc[-1] is not np.nan else np.nan
                    # sometimes the first few rows of a trial_df, when grouped by trial_index, have the previous condition
                    new_trial_row['condition'] = trial_df_filtered_with_condition['conditionType'].mode().item()

                    deduplicated_user_inputs = trial_df_filtered_with_condition['currentText'].drop_duplicates(keep='first')
                    try:
                        np.isnan(deduplicated_user_inputs.iloc[0])
                    except TypeError as e:
                        raise e
                    if len(deduplicated_user_inputs) == 1 and np.isnan(deduplicated_user_inputs.iloc[0]):
                        warnings.warn(f"trial {trial_index} for Participant {participant_id} session {session_number} has no input."
                                      f"Deduplicated user input text are"
                                      f"{deduplicated_user_inputs}. \n"
                                      f"Setting this trial to invalid.")
                        new_trial_row['isValid'] = 'No user input'
                        trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
                        continue

                    # TODO if the levenshtein distance between the target sentence and the last user input is too great, drop the trial
                    new_trial_row['FinalUserInputEditDistance2Target'] = nltk.edit_distance(user_inputs_step.iloc[-1], target_sentence)
                    new_trial_row['FinalUserInputEditDistance2TargetNormalized'] = new_trial_row['FinalUserInputEditDistance2Target'] / max(len(user_inputs_step.iloc[-1]), len(target_sentence))

                    # find the first input
                    if new_trial_row['condition'] in swype_conditions:
                        first_input_index = trial_df_filtered_with_condition[trial_df_filtered_with_condition['xKeyHitLocal'] != -np.inf].index[0]  # this index is that of action_info_df's instead of trial_df's
                    elif new_trial_row['condition'] == 'HandTap' or new_trial_row['condition'] == 'GazePinch':
                        first_input_index = 0
                        if 'eyeTrackingStatus' not in trial_df_filtered_with_condition.columns:  # need to check if the first ActionInfo input index is within the bounds of the stream timestamps
                            _, diff = find_closes_time_index(eyetracking_stream_timestamps, trial_df_filtered_with_condition['absoluteTime'].iloc[first_input_index], return_diff=True)
                            if abs(diff) >= 5e-3 :
                                first_input_index = trial_df_filtered_with_condition[np.logical_not(pd.isna(trial_df_filtered_with_condition['keyboardValue']))].index[0]  # this index is that of action_info_df's instead of trial_df's
                        if first_input_index == 0: trial_first_index_is_zero_counter += 1
                    else:
                        raise Exception(f"Unknown condition: {new_trial_row['condition']}")
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
                    trial_timestamps = np.array(trial_w_input_df['absoluteTime'].values, dtype=float)

                    if new_trial_row['condition'] in eyetracking_conditions:  # removing the calibration time
                    # if new_trial_row['condition'] == 'GazePinch':  # removing the calibration time
                        # remove the rows of calibration
                        # there are two ways to know if is calibrating,
                        # * one is using the column 'eyeTrackingStatus'
                        if 'eyeTrackingStatus' in trial_w_input_df.columns:
                            duration = trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0]
                            if len(trial_w_input_df[trial_w_input_df['eyeTrackingStatus'] != 'Available']) > 0:
                                eye_status_trial = np.array([1 if x == 'Available' else 0 for x in trial_w_input_df['eyeTrackingStatus']])
                                try:
                                    invalid_duration = compute_invalid_duration(eye_status_trial, trial_timestamps)
                                except Exception as e:
                                    raise e
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

                    if new_trial_row['condition'] in swype_conditions:
                        eye_status_col_name = 'eyeTrackingStatus' if 'eyeTrackingStatus' in trial_w_input_df.columns else 'eyeTrackingStatusFittedFromStream'
                        assert eye_status_col_name in trial_w_input_df.columns, f"Participant {participant_id} session {session_number} trial {trial_index} does not have the column {eye_status_col_name}."

                        key_hit = trial_w_input_df['xKeyHitLocal'].values
                        candidate1_is_na = pd.isna(trial_w_input_df['candidate1']).values
                        eye_status = trial_w_input_df[eye_status_col_name].values
                        try:
                            eye_hit_status = np.array([(0 if abs(x_key_hit) == np.inf and x_cndt_is_na else 1) for x_key_hit, x_cndt_is_na in zip(key_hit, candidate1_is_na)])
                        except TypeError as e:
                            raise e
                        hit_invalid_duration = compute_invalid_duration(eye_hit_status, trial_timestamps)

                        eye_hit_calib_status = np.array([(0 if ((abs(x_key_hit) == np.inf and x_cndt_is_na ) or x_status == 0 )else 1) for x_key_hit, x_cndt_is_na, x_status in zip(key_hit, candidate1_is_na, eye_status)])
                        hit_calib_invalid_duration = compute_invalid_duration(eye_hit_calib_status, trial_timestamps)

                        # print(f"Total duration: {trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0]}. Hit ID: {hit_invalid_duration}. Hit calib ID: {hit_calib_invalid_duration}")
                        invalid_duration = hit_calib_invalid_duration
                    else:
                        invalid_duration = 0

                    new_trial_row['duration'] = trial_w_input_df['trialTime'].iloc[-1] - trial_w_input_df['trialTime'].iloc[0] - invalid_duration  # trial_df at this point is already cleaned
                    new_trial_row['wpm'] = (60/5) * len(new_trial_row['userSentence']) / new_trial_row['duration']
                    # new_trial_row['wpm'] = count_words(new_trial_row['userSentence']) / new_trial_row['duration'] * 60

                    # 1. overall CPM, CPM as a function word length
                    new_trial_row['overallCPM'] = len(new_trial_row['userSentence']) / new_trial_row['duration'] * 60

                    # find all the words in the final sentence
                    words = re.findall(r'\b\w+\b', new_trial_row['userSentence'])
                    # replace nan in currentText with empty string
                    trial_w_input_df = trial_w_input_df.copy()
                    trial_w_input_df.loc[pd.isna(trial_w_input_df['currentText']), 'currentText'] = ""
                    trial_w_input_df.loc[:, 'currentWords'] = trial_w_input_df['currentText'].apply(lambda x: re.findall(r'\b\w+\b', x))  # TODO these three lines comes out with warnings
                    trial_w_input_df.loc[:, 'xKeyHitLocalValid'] = trial_w_input_df['xKeyHitLocal'].apply(lambda x: abs(x) != np.inf)
                    trial_w_input_df.loc[:, 'nWords'] = trial_w_input_df.copy()['currentWords'].apply(lambda x: len(x))

                    sweyepe_durations = []
                    w_first_appear_index = trial_w_input_df.index[0]
                    for w in words:
                        if len(w) == 1:
                            continue  # ignore single letter
                        # find the index when w first appear in trial_w_input_df['currentText']
                        # in case a word appearing multiple times in the currentText,
                        subset_df = trial_w_input_df.loc[w_first_appear_index:]
                        subset_df = subset_df[subset_df['currentWords'].apply(lambda x: w in x)]
                        w_first_appear_index = subset_df.index[0]

                        # w_first_appear_index = trial_w_input_df.loc[w_first_appear_index:][trial_w_input_df['currentWords'].apply(lambda x: w in x)].index[0]
                        if new_trial_row['condition'] not in swype_conditions:
                            # find the last time nWords increment∂ in the series up to w_first_appear_index, this is the beginning letter of the new word first being typed
                            nWords_last_increment_index = trial_w_input_df.loc[:w_first_appear_index, 'nWords'].idxmax()
                            assert w_first_appear_index >= nWords_last_increment_index, f"Participant {participant_id} session {session_number} trial {trial_index} word {w} first appear index is smaller than the last nWords increment index."
                            w_duration = trial_w_input_df['absoluteTime'][w_first_appear_index] - trial_w_input_df['absoluteTime'][nWords_last_increment_index]
                        else:  # in Sweyepe, the start-typing time for a word is when the sweyepe trace starts
                            hit_state_transition = np.argwhere(np.diff(np.concatenate([[False], trial_w_input_df.loc[:w_first_appear_index, 'xKeyHitLocalValid'].values])))
                            try:
                                sweyepe_start_index = trial_w_input_df.index[hit_state_transition[-2][0]]
                            except IndexError as e:
                                print(e)
                            sweyepe_end_index = trial_w_input_df.index[hit_state_transition[-1][0]]
                            try:
                                assert w_first_appear_index >= sweyepe_start_index
                            except:
                                raise Exception(f"Participant {participant_id} session {session_number} trial {trial_index} word {w} first appear index is smaller than the sweyepe start index.")
                            assert sweyepe_end_index >= sweyepe_start_index
                            w_duration = trial_w_input_df['absoluteTime'][sweyepe_end_index] - trial_w_input_df['absoluteTime'][sweyepe_start_index]
                            sweyepe_durations.append(w_duration)
                            # 3. Sweyepe first candidate miss rate, all candidate miss rate
                            # find the first row after sweyepe_end_index, AND candidates are available, AND the current text has content
                            # THE CURRENT TEXT MUST HAVE CONTENT because the candidates from previous trial may be carried over to the next trial, we need to find the first row where the currentText is not empty
                            n_sweyepe_words += 1
                            first_candidate_available_index = ~pd.isna(trial_w_input_df.loc[sweyepe_end_index:, 'candidate1']) & (trial_w_input_df.loc[sweyepe_end_index:, 'currentText'] != "") & (trial_w_input_df.loc[sweyepe_end_index:, 'currentText'].apply(lambda x: not x.isspace()))
                            first_candidate_available_index = first_candidate_available_index.idxmax()

                            cur_words = [x.lower() for x in re.findall(r'\b\w+\b', trial_w_input_df.loc[first_candidate_available_index]['currentText'])]
                            try:
                                first_candidate = cur_words[-1]
                            except IndexError:
                                first_candidate = None
                            candidate123 = [trial_w_input_df.loc[first_candidate_available_index][f'candidate{i}'] for i in range(1, 4)]
                            # target word is the word the participant is trying to type
                            # the target word is first word in target_sentence after removing the words that are already in cur_words
                            words_in_this_trial = [x.lower() for x in re.findall(r'\b\w+\b', target_sentence)]
                            already_typed_words = cur_words[:-1]  # already typed words up to the last word
                            for tw in already_typed_words:
                                if tw in words_in_this_trial:
                                    words_in_this_trial.remove(tw)
                            try:
                                target_word = words_in_this_trial[0]
                            except IndexError as e:
                                print(f"{e}: words_in_this_trial doesn't have any words left after removing already typed words.")

                            if first_candidate == target_word:
                                n_sweyepe_first_candidate_matched += 1
                                n_sweyepe_any_candidates_matched += 1
                            elif target_word in candidate123:
                                n_sweyepe_any_candidates_matched += 1
                            else:
                                n_sweyepe_all_candidate_missed += 1
                        if f'CPMWordLen={len(w)}' not in new_trial_row:
                            new_trial_row[f'CPMWordLen={len(w)}'] = [len(w) / w_duration * 60]
                        else:
                            try:
                                new_trial_row[f'CPMWordLen={len(w)}'] = new_trial_row[f'CPMWordLen={len(w)}'] + [len(w) / w_duration * 60]
                            except TypeError as e:
                                raise e
                        # new_trial_row[f'CPMWordLen={len(w)}'] = [len(w) / w_duration * 60] if f'CPMWordLen={len(w)}' not in new_trial_row else new_trial_row[f'CPMWordLen={len(w)}'] + [len(w) / w_duration * 60]
                    if new_trial_row['condition'] in swype_conditions:
                        if np.any(np.isnan(sweyepe_durations)):
                            print(f"found NAN in sweyepe durations: {sweyepe_durations}")
                        new_trial_row['average_sweyepe_duration'] = np.mean(sweyepe_durations)

                    # 3. Sweyepe first candidate miss rate, all candidate miss rate
                    if new_trial_row['condition'] in swype_conditions:
                        try:
                            new_trial_row['sweyepe_first_candidate_match_rate'] = n_sweyepe_first_candidate_matched / n_sweyepe_words
                            new_trial_row['sweyepe_any_candidate_match_rate'] = n_sweyepe_any_candidates_matched / n_sweyepe_words
                            new_trial_row['sweyepe_all_candidate_miss_rate'] = n_sweyepe_all_candidate_missed / n_sweyepe_words
                        except ZeroDivisionError:
                            new_trial_row['sweyepe_first_candidate_match_rate'] = np.nan
                            new_trial_row['sweyepe_any_candidate_match_rate'] = np.nan
                            new_trial_row['sweyepe_all_candidate_miss_rate'] = np.nan
                    # compute the average CPM for each word length
                    for i in range(1, 30):
                        if f'CPMWordLen={i}' in new_trial_row:
                            new_trial_row[f'CPMWordLen={i}'] = np.mean(new_trial_row[f'CPMWordLen={i}'])
                    # only count non-consecutive backspace
                    delete_presses = (trial_w_input_df["keyboardValue"] == "Delete") & (trial_w_input_df["keyboardValue"].shift() != "Delete")
                    if new_trial_row['condition'] in swype_conditions:
                        new_trial_row['numDeletePress'] = new_trial_row['sweyepe_all_candidate_miss_rate']
                    else:
                        new_trial_row['numDeletePress'] = delete_presses.sum()
                    new_trial_row['numDeletePressPerChar'] = new_trial_row['numDeletePress'] / len(new_trial_row['userSentence'])

                    # 3. Compute TER
                    # first get the input stream
                    unique_cur_text = [x.lower() for x in pd.unique(trial_w_input_df.loc[:, 'currentText'])]
                    input_stream = reconstruct_input_stream(unique_cur_text)
                    new_trial_row['TER'] = compute_total_error_rates(target_sentence.lower(), input_stream, new_trial_row['userSentence'].lower())["Total Error Rate (%)"]
                    new_trial_row['MSD Error Rate'] = compute_msd_error_rates(target_sentence.lower(), new_trial_row['userSentence'].lower())

                    if new_trial_row['isValid']:
                        condition_trial_index[new_trial_row['condition']] += 1
                        new_trial_row['condition_trial_index'] = condition_trial_index[new_trial_row['condition']]  # the index of the trial for this condition
                        if new_trial_row['condition_trial_index'] > 16:
                            warnings.warn(f"!!!!!! Participant {participant_id} session {session_number} trial {trial_index} has more than 12 trials for this condition. It has {new_trial_row['condition_trial_index']} trials \n")

                    # action_info_df.iloc[last_unique_text_index]['trialTime'] - action_info_df.iloc[first_input_index]['trialTime']
                    # report the number of valid trials for each conditions that this session has

                    # TODO check the number of completion and next word prediction usage when is study 2

                    # TODO check the number of partial sweyepe completes for PartialSwipe and HandSwipe
                    # pending on Season's response

                    # TODO number of deletes mid swipe for PartialSwipe and HandSwipe

                    trial_data = pd.concat([trial_data, pd.DataFrame([new_trial_row])], ignore_index=True)
                    trial_dfs.append(trial_df_filtered_with_condition)

        # save the trial_data
        trial_data.to_csv(trial_data_export_path, index=False)
    else:
        trial_data = pd.read_csv(trial_data_export_path)

# plot the wpm as a function of session ##############################################################
assert trial_data.shape[0] > 0, "trial data is empty"

trial_data_wpm_na_dropped = trial_data.dropna(subset=['wpm'])
sns.boxplot(data=trial_data_wpm_na_dropped, x="session", y="wpm", hue="condition")
plt.ylim(0, 60)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Adds a dashed grid
plt.show()

sns.boxplot(data=trial_data_wpm_na_dropped, x="session", y="overallCPM", hue="condition")
plt.ylim(0, 200)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Adds a dashed grid
plt.show()

sns.catplot(data=trial_data_wpm_na_dropped, x="session", y="TER", hue="condition", kind="bar")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Adds a dashed grid
plt.ylabel("Total Error Rate (%)")
plt.show()

sns.catplot(data=trial_data_wpm_na_dropped, x="session", y="MSD Error Rate", hue="condition", kind="bar")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Adds a dashed grid
plt.ylabel("MSD Error Rate (%)")
plt.show()


# plot the performance as a function of trials in the users's first session
trial_data_first_session = trial_data[trial_data['session'] == 1]
trial_data_first_session = trial_data_first_session.dropna(subset=['wpm'])
trial_data_first_session = trial_data_first_session[trial_data_first_session['condition_trial_index'] <= 12]
sns.lineplot(data=trial_data_first_session, x="condition_trial_index", y="wpm", hue="condition", style="condition", markers=True, err_style="bars", errorbar=("se", 2))
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Adds a dashed grid
plt.show()







# backspace as a function of session ##############################################################
sns.lineplot(data=trial_data_wpm_na_dropped, x="session", y="numDeletePressPerChar", hue="condition", style="condition", markers=True,err_style="bars", errorbar=("se", 2),)
# plt.ylim(0, 60)
plt.show()

sns.catplot(data=trial_data_wpm_na_dropped, x="session", y="numDeletePressPerChar", hue="condition")
plt.show()


# cpm as a function of word length, separate plots for each condition  ##############################################################

df = trial_data_wpm_na_dropped.melt(
    id_vars=[col for col in trial_data_wpm_na_dropped.columns if not col.startswith('CPMWordLen')],
    value_vars=[col for col in trial_data_wpm_na_dropped.columns if col.startswith('CPMWordLen')],
    var_name='word_len',
    value_name='CPM_by_word_len'
)
# Drop rows where CPM_by_word_len is NaN
df = df.dropna(subset=['CPM_by_word_len'])
df['word_len'] = df['word_len'].str.extract('(\d+)').astype(int)

Q1 = df["CPM_by_word_len"].quantile(0.25)
Q3 = df["CPM_by_word_len"].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
# Identify rows to drop
extreme_values = df[(df["CPM_by_word_len"] < lower_bound) | (df["CPM_by_word_len"] > upper_bound)]
df_filtered = df[(df["CPM_by_word_len"] >= lower_bound) & (df["CPM_by_word_len"] <= upper_bound)]

g = sns.catplot(
    data=df_filtered[df_filtered['word_len'] <= 12], x="word_len", y="CPM_by_word_len", hue="condition", col="session",
    capsize=.2, palette="YlGnBu_d", errorbar="se",
    kind="point", height=6, aspect=.75,
)
g.despine(left=True)
plt.ylim(0, 300)
plt.show()


# plot the sweyepe miss rate  ##############################################################
"""
construct a dataframe for sweyepe miss rate: like this

session     trial   type    rate
1           1       first   0.2
1           1       any     0.5
1           1       all     0.3
1           2       first   0.1
1           2       any     0.2
1           2       all     0.3
...

"""
sweyepe_miss_rate = trial_data.dropna(subset=['sweyepe_first_candidate_match_rate', 'sweyepe_any_candidate_match_rate', 'sweyepe_all_candidate_miss_rate'])
sweyepe_miss_rate = sweyepe_miss_rate[['session', 'trial_index', 'sweyepe_first_candidate_match_rate', 'sweyepe_any_candidate_match_rate', 'sweyepe_all_candidate_miss_rate']]
sweyepe_miss_rate = pd.melt(sweyepe_miss_rate, id_vars=['session', 'trial_index'], var_name='type', value_name='rate')
"""
Rename the type to more meaningful names
sweyepe_first_candidate_match_rate -> First Candidate Match
sweyepe_any_candidate_match_rate -> Any Candidate Match
sweyepe_all_candidate_miss_rate -> All Candidate Miss
"""
name_map = {
    'sweyepe_first_candidate_match_rate': 'First Candidate Match',
    'sweyepe_any_candidate_match_rate': 'Any Candidate Match',
    'sweyepe_all_candidate_miss_rate': 'All Candidate Miss'
}
sweyepe_miss_rate['type'] = sweyepe_miss_rate['type'].map(name_map)
sweyepe_miss_rate["session"] = pd.Categorical(
    sweyepe_miss_rate["session"], categories=sorted(sweyepe_miss_rate["session"].unique()), ordered=True
)

g = sns.catplot(data=sweyepe_miss_rate, x="session", y="rate", hue="type", kind="bar")
g.fig.set_figwidth(15)
g.fig.set_figheight(6)
ax = g.ax
# Overlay a line plot
# g = sns.lineplot(
#     data=sweyepe_miss_rate, x="session", y="rate", hue="type", marker="o", ax=ax
# )
for p in ax.patches:
    if p.get_height() > 0:
        ax.text(
            p.get_x() + p.get_width() / 2,  # x-coordinate (center of the bar)
            p.get_height() + 0.05,  # y-coordinate (top of the bar)
            f'{p.get_height():.3f}',  # Label text formatted to two decimal places
            ha='center',  # Centered horizontally
            va='bottom',  # Positioned just above the bar
            fontsize=10  # Adjust font size
        )
plt.show()





