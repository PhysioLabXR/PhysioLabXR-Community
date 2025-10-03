import pickle
import numpy as np
import pandas as pd
from collections import deque, OrderedDict
from itertools import groupby
from physiolabxr.scripting.illumiRead.utils.VarjoEyeTrackingUtils.VarjoGazeUtils import VarjoGazeData
from physiolabxr.scripting.illumiRead.utils.gaze_utils.general import GazeFilterFixationDetectionIVT, GazeType
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeUtils import illumiReadSwypeUserInput
import csv
import re
# Keyboard proximity map for horizontally adjacent keys
keyboard_proximity = {
    'q': 'qw', 'w': 'we', 'e': 'er', 'r': 'rt', 't': 'ty',
    'y': 'yu', 'u': 'ui', 'i': 'io', 'o': 'op', 'p': 'o',
    'a': 'as', 's': 'ad', 'd': 'sf', 'f': 'dg', 'g': 'fh',
    'h': 'gj', 'j': 'hk', 'k': 'jl', 'l': 'k',
    'z': 'zx', 'x': 'zc', 'c': 'xv', 'v': 'cb', 'b': 'vn',
    'n': 'bm', 'm': 'n'
}

# Helper function to check if two letters are close based on the proximity map
def matches_with_proximity(letter, possible_letters):
    """Check if a letter or any of its close neighbors are in possible_letters."""
    neighbors = keyboard_proximity.get(letter, '') + letter
    return any(l in possible_letters for l in neighbors)

# Helper function to check if two consecutive letters in target text can be parsed together
def parse_two_with_proximity(current_letter, next_letter, possible_letters):
    """Check if current and next letter match the fixation with proximity."""
    if current_letter in possible_letters and next_letter in possible_letters:
        if next_letter in keyboard_proximity.get(current_letter, ''):
            return True
    return False
def parse_tuple(val):
    """Convert a string tuple like '(-0.5332503 0.01100001)' to a Python tuple."""
    return tuple(map(float, val.strip('()').split()))


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
    pickle_file_path = 'practice.p'  # Path to pickle file containing LSL data
    with open(pickle_file_path, 'rb') as file:
        data = pickle.load(file)

    # Extract Gaze and User Input Data
    gaze_channels = data['VarjoEyeTrackingLSL'][0]  # Shape: (39, N)
    gaze_timestamps = data['VarjoEyeTrackingLSL'][1]  # Shape: (N,)
    user_input_data = data['illumiReadSwypeUserInputLSL']  # [0]: data (11, M), [1]: timestamps (M,)
    # Load ActionInfo CSV and Determine Sweyepe Start Time
    action_info_path = 'ActionInfo.csv'  # Path to ActionInfo.csv
    action_info = pd.read_csv(action_info_path)

    # Extract target text for Sweyepe events (unique sentences only)
    sweyepe_texts = action_info.loc[action_info['conditionType'] == 'Sweyepe', 'targetText'].drop_duplicates()
    sweyepe_texts = [list(target_text.replace(" ", "").lower()) for target_text in sweyepe_texts]

    # Determine Sweyepe Start Time
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

    # Fixation and letter mappings storage
    fixation_to_letters = []

    # Merge and Sort Streams by Timestamp (Filter by Sweyepe Start Time)
    all_data_stream = []
    for i, timestamp in enumerate(gaze_timestamps):
        if timestamp >= sweyepe_start_time:
            all_data_stream.append(('gaze', i, timestamp))
    for i, timestamp in enumerate(user_input_data[1]):
        if timestamp >= sweyepe_start_time:
            all_data_stream.append(('user_input', i, timestamp))
    all_data_stream.sort(key=lambda x: x[2])  # Sort by timestamp
    fixation_summary = []  # List to store fixation times and possible letters
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
                # Extract fixation start and end times
                fixation_start = fixation_points[0][0]
                fixation_end = fixation_points[-1][0]

                # Map Fixation to Letters
                possible_letters = map_fixation_to_letters(
                    fixation_points, user_input_data, letter_locations, radius=0.2
                )
                if possible_letters:
                    fixation_to_letters.append({"fixation_points": fixation_points, "letters": possible_letters})
                    # Save fixation summary with times
                    fixation_summary.append({
                        "fixation_start": fixation_start,
                        "fixation_end": fixation_end,
                        "possible_letters": possible_letters
                    })

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

    # Save results for further processing
    # print("Fixation Summary:")
    # for summary in fixation_summary:
    #     print(f"Fixation Start: {summary['fixation_start']}, Fixation End: {summary['fixation_end']}")
    #     print(f"Possible Letters: {summary['possible_letters']}")
    #     print()
    # print("Target Texts:", sweyepe_texts)
    # Initialize result list
    # Initialize result list
    # Main processing logic
    fixation_results = []  # Store fixation data with marking

    # Track current position in the sentence
    sentence_position = 0

    # Convert sweyepe_texts to a target_text (flattened list of characters)
    sweyepe_texts = action_info.loc[action_info['conditionType'] == 'Sweyepe', 'targetText'].drop_duplicates()
    sweyepe_texts = [list(target_text.replace(" ", "").lower()) for target_text in sweyepe_texts]
    target_text = [char for sentence in sweyepe_texts for char in sentence]

    # Process each fixation in the summary
    for fixation in fixation_summary:
        fixation_start = fixation['fixation_start']
        fixation_end = fixation['fixation_end']
        possible_letters = fixation['possible_letters']

        # Ensure we have a valid current letter
        if sentence_position >= len(target_text):
            break

        current_letter = target_text[sentence_position]
        next_letter = target_text[sentence_position + 1] if sentence_position + 1 < len(target_text) else None

        # Check if the current fixation matches the current letter
        if current_letter in possible_letters:
            # Mark this fixation as 1
            fixation_results.append({
                "fixation_start": fixation_start,
                "fixation_end": fixation_end,
                "possible_letters": possible_letters,
                "current_letter": current_letter,
                "mark": 2
            })

            # Parse all consecutive identical letters in the target
            while sentence_position + 1 < len(target_text) and target_text[sentence_position + 1] == current_letter:
                sentence_position += 1
                fixation_results.append({
                    "fixation_start": fixation_start,
                    "fixation_end": fixation_end,
                    "possible_letters": possible_letters,
                    "current_letter": current_letter,
                    "mark": 2
                })

            # Check if the next target letter can also be parsed with the current fixation
            if next_letter and parse_two_with_proximity(current_letter, next_letter, possible_letters):
                fixation_results.append({
                    "fixation_start": fixation_start,
                    "fixation_end": fixation_end,
                    "possible_letters": possible_letters,
                    "current_letter": next_letter,
                    "mark": 2
                })
                sentence_position += 1  # Skip the next letter

            # Move to the next target letter after handling all consecutive ones
            sentence_position += 1

        else:
            # Mark this fixation as 0 (current letter not found in possible letters)
            fixation_results.append({
                "fixation_start": fixation_start,
                "fixation_end": fixation_end,
                "possible_letters": possible_letters,
                "current_letter": current_letter,
                "mark": 1
            })

            # If the next target letter matches, advance the sentence position
            if next_letter and matches_with_proximity(next_letter, possible_letters):
                sentence_position += 1

    # Final Results
    # print("Final Fixation Results:")
    # for res in fixation_results:
    #     print(f"Fixation Start: {res['fixation_start']}, Fixation End: {res['fixation_end']}, "
    #           f"Possible Letters: {res['possible_letters']}, Current Letter: {res['current_letter']}, Mark: {res['mark']}")

    # Extract DSI24 data from the pickle file
    dsi_data = data['DSI24']
    dsi_channels = dsi_data[0]  # Shape: (24, n)
    dsi_timestamps = dsi_data[1]  # Shape: (n,)

    # Filter DSI data based on fixation markings
    filtered_dsi_results = []

    # Iterate over fixation results to mark DSI data
    for fixation in fixation_results:
        fixation_start = fixation['fixation_start']
        fixation_end = fixation['fixation_end']
        mark = fixation['mark']  # 0, 1, or 2

        # Filter DSI timestamps within the fixation start and end range
        dsi_indices = (dsi_timestamps >= fixation_start) & (dsi_timestamps <= fixation_end)
        if dsi_indices.any():
            filtered_dsi_results.append({
                "start": dsi_timestamps[dsi_indices][0],
                "end": dsi_timestamps[dsi_indices][-1],
                "mark": mark
            })

    # Add non-fixation times (mark 0)
    # Determine gaps in DSI data that are outside any fixation time
    fixation_time_ranges = [(fix['fixation_start'], fix['fixation_end']) for fix in fixation_results]
    non_fixation_times = []

    previous_end = dsi_timestamps[0]  # Start of DSI data
    for start, end in fixation_time_ranges:
        if previous_end < start:
            non_fixation_indices = (dsi_timestamps >= previous_end) & (dsi_timestamps < start)
            if non_fixation_indices.any():
                non_fixation_times.append({
                    "start": dsi_timestamps[non_fixation_indices][0],
                    "end": dsi_timestamps[non_fixation_indices][-1],
                    "mark": 0
                })
        previous_end = end

    # Add the non-fixation times to filtered results
    filtered_dsi_results.extend(non_fixation_times)

    # Sort all results by start time to maintain chronological order
    filtered_dsi_results.sort(key=lambda x: x['start'])

    # Output the DSI filtering results
    print("Filtered DSI Results:")
    for res in filtered_dsi_results:
        print(f"Start: {res['start']}, End: {res['end']}, Mark: {res['mark']}")
    # Save filtered DSI results to a CSV file
    csv_file_path = 'filtered_dsi_results.csv'
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Start', 'End', 'Mark'])

        # Write the data
        for res in filtered_dsi_results:
            writer.writerow([res['start'], res['end'], res['mark']])

    print(f"Filtered DSI results saved to {csv_file_path}")
