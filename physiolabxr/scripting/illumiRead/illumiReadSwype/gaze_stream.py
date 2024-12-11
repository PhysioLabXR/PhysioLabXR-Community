import pickle
import numpy as np
import pandas as pd
from collections import deque, OrderedDict
from itertools import groupby
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeUtils import VarjoGazeData
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeFilterFixationDetectionIVT, GazeType
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeUtils import illumiReadSwypeUserInput

import re


def parse_tuple(val):
    """Convert a string tuple like '(-0.5332503 0.01100001)' to a Python tuple."""
    return tuple(map(float, val.strip('()').split()))
# Helper Function: Parse Letter Locations
# Helper Function: Parse Letter Locations
def parse_letter_locations(gaze_data_path):
    """Parse key ground truth locations from the CSV."""
    letters = []
    key_ground_truth_local = []

    with open(gaze_data_path, 'r') as file:
        header = file.readline()  # Skip header row
        for line in file:
            if not line.strip() or line.startswith('Key'):  # Skip invalid lines
                continue
            if re.match(r'^[a-zA-Z]', line):  # Match valid letter rows
                parts = line.strip().split(',')
                letters.append(parts[0])  # Letter
                key_ground_truth_local.append(parse_tuple(parts[3]))  # KeyGroundTruthLocal column

    # Group by letter and calculate mean location
    df = pd.DataFrame({'Letter': letters, 'KeyGroundTruthLocal': key_ground_truth_local})
    grouped = df.groupby('Letter')['KeyGroundTruthLocal']
    letter_locations = OrderedDict()
    for letter, group in grouped:
        ground_truth_array = np.array(list(group))
        letter_locations[letter] = np.mean(ground_truth_array, axis=0)
    return letter_locations



# Function to Map Fixations to Letters
def map_fixation_to_letters(fixation_points, user_input_data, letter_locations, radius=0.05):
    """Map fixation points to possible letters using user input hit points and letter locations."""
    if not fixation_points:
        return []

    # Calculate fixation centroid
    fixation_array = np.array([[point[1][0], point[1][1]] for point in fixation_points])
    centroid = np.mean(fixation_array, axis=0)

    # Filter user input data within fixation timestamps
    fixation_start = fixation_points[0][0]
    fixation_end = fixation_points[-1][0]
    user_input_in_fixation = [
        illumiReadSwypeUserInput(data, timestamp)
        for data, timestamp in zip(user_input_data[0].T, user_input_data[1])
        if fixation_start <= timestamp <= fixation_end
    ]

    # Get keyboard hit points from user input
    keyboard_hit_points = [
        np.array(input_data.keyboard_background_hit_point_local[:2])
        for input_data in user_input_in_fixation
        if not np.array_equal(input_data.keyboard_background_hit_point_local, [1, 1, 1])
    ]

    # Map fixation centroid to letters
    possible_letters = []
    for hit_point in keyboard_hit_points:
        for letter, location in letter_locations.items():
            distance = np.linalg.norm(location - hit_point)
            if distance <= radius:
                possible_letters.append(letter)
    return list(set(possible_letters))



# Main Script
if __name__ == "__main__":
    # Load Varjo Eye and User Input Data
    pickle_file_path = 'hardexperiment.p'  # Path to pickle file containing LSL data
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract Gaze and User Input Data
    gaze_channels = data['VarjoEyeTrackingLSL'][0]  # Shape: (39, N)
    gaze_timestamps = data['VarjoEyeTrackingLSL'][1]  # Shape: (N,)
    user_input_data = data['illumiReadSwypeUserInputLSL']  # [0]: data (11, M), [1]: timestamps (M,)
    # Load ActionInfo CSV and Determine Sweyepe Start Time
    action_info_path = 'ActionInfo-1.csv'  # Path to ActionInfo.csv
    action_info = pd.read_csv(action_info_path)
    sweyepe_start_time = action_info.loc[action_info['conditionType'] == 'Sweyepe', 'absoluteTime'].min()
    if pd.isna(sweyepe_start_time):
        raise ValueError("No sweyepe mode found in ActionInfo.csv")
    # Path to Gaze Data CSV for Letter Locations
    gaze_data_csv_path = 'GazeData.csv'  # Path to gaze data file
    letter_locations = parse_letter_locations(gaze_data_csv_path)

    # Initialize IVT Filter for Fixation Detection
    ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)
    gaze_data_sequence = []
    fixation_points_buffer = []

    # Merge and Sort Streams by Timestamp (Filter by Sweyepe Start Time)
    all_data_stream = []
    for i, timestamp in enumerate(gaze_timestamps):
        if timestamp >= sweyepe_start_time:
            all_data_stream.append(('gaze', i, timestamp))
    for i, timestamp in enumerate(user_input_data[1]):
        if timestamp >= sweyepe_start_time:
            all_data_stream.append(('user_input', i, timestamp))
    all_data_stream.sort(key=lambda x: x[2])  # Sort by timestamp

    # Simulate Real-Time Processing
    for data_type, index, timestamp in all_data_stream:
        if data_type == 'gaze':
            # Process Gaze Data
            gaze_sample = gaze_channels[:, index]

            # Create VarjoGazeData Object
            gaze_data = VarjoGazeData()
            gaze_data.construct_gaze_data_varjo(gaze_sample, timestamp)

            # Apply IVT Filter
            processed_gaze_data = ivt_filter.process_sample(gaze_data)
            gaze_data_sequence.append(processed_gaze_data)

            # Check for Fixation End
            if processed_gaze_data.gaze_type != GazeType.FIXATION and fixation_points_buffer:
                fixation_points = fixation_points_buffer.copy()
                fixation_points_buffer = []  # Reset buffer

                # Map Fixation to Letters
                possible_letters = map_fixation_to_letters(
                    fixation_points, user_input_data, letter_locations, radius=0.2
                )
                if possible_letters:
                    print(f"Fixation: {fixation_points}")
                    # print(user_input_data)
                    print(f"Possible Letters: {possible_letters}")

            # If Fixation, Add to Buffer
            if processed_gaze_data.gaze_type == GazeType.FIXATION:
                fixation_points_buffer.append(
                    (processed_gaze_data.timestamp, processed_gaze_data.get_combined_eye_gaze_direction()[:2])
                )

        elif data_type == 'user_input':
            # Process User Input Data
            user_input_sample = user_input_data[0][:, index]
            user_input_timestamp = user_input_data[1][index]

            # Check for Button Input (e.g., Start/End Sweyepe)
            user_input = illumiReadSwypeUserInput(user_input_sample, user_input_timestamp)
            if user_input.user_input_button_2:
                pass
                # print(f"Sweyepe Event Detected at {user_input_timestamp}")
