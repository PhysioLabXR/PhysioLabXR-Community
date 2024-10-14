import re
from collections import OrderedDict

import numpy as np
import pandas as pd
from nltk import TreebankWordDetokenizer
from sklearn.cluster import DBSCAN


def parse_tuple(val):
    # Remove the parentheses and split the string into individual components
    return tuple(map(float, val.strip('()').split()))


def load_trace_file(file_path):
    trace = pd.read_csv(file_path)

    # Initialize the list to hold arrays
    trace_list = []

    # Temporary list to hold current array
    current_array = []

    # Iterate over the rows of the dataframe
    for index, row in trace.iterrows():
        if pd.isna(row['KeyBoardLocalY']):
            # If the row contains only one element, start a new array
            if current_array:
                trace_list.append(np.array(current_array).astype(float))
                current_array = []
        else:
            # Append the row to the current array
            current_array.append(row.tolist())

    return trace_list

def parse_letter_locations(gaze_data_path):
    letters = []
    hit_point_local = []
    key_ground_truth_local = []

    # Read and process the lines
    with open(gaze_data_path, 'r') as file:
        header = file.readline()  # Read the header
        # skip the first line and read the rest
        for i, line in enumerate(file):
            if re.match(r'^[a-zA-Z]', line):
                parts = line.strip().split(',')
                letters.append(parts[0])  # Extract the letter
                hit_point_local.append(parse_tuple(parts[2]))
                key_ground_truth_local.append(parse_tuple(parts[3]))

    # Create a DataFrame from the extracted data
    df = pd.DataFrame({
        'Letter': letters,
        'HitPointLocal': hit_point_local,
        'KeyGroundTruthLocal': key_ground_truth_local
    })

    # for each unique letter in the letter column, get their KeyGroundTruthLocal put them in a 2d array
    grouped = df.groupby('Letter')['KeyGroundTruthLocal']
    letter_locations = OrderedDict()
    for letter, group in grouped:
        # Convert the 'KeyGroundTruthLocal' tuples to a 2D array
        ground_truth_array = np.array(list(group))
        assert np.all(np.std(ground_truth_array, axis=0) < np.array([5e-6,
                                                                     5e-6])), f"std of the groundtruth is not close to zero, letter is {letter}, std is {np.std(ground_truth_array, axis=0)}"
        letter_locations[letter] = np.mean(ground_truth_array, axis=0)

    return letter_locations

def run_dbscan_on_gaze(gaze_trace, timestamps, dbscan_eps, dbscan_min_samples, verbose):
    """Run DBSCAN on the gaze trace to reduce the number of points.

    If DBSCAN finds zero clusters, return the original gaze trace.

    Args:
        gaze_trace: ndarray (2, t): The gaze trace to reduce.
        timestamps: ndarray (t,): The timestamps for each gaze point.
        dbscan_eps: float: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        dbscan_min_samples: int: The number of samples in a neighborhood for a point to be considered as a core point.
        verbose: bool: Whether to print verbose output.

    Returns:
        ndarray (n, 2): The reduced gaze trace.

    """
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)

    if timestamps is not None:
        assert len(timestamps) == len(gaze_trace)
        dbscan.fit(np.concatenate([gaze_trace, timestamps[..., None]], axis=-1))
    else:
        dbscan.fit(gaze_trace)
    labels = dbscan.labels_

    # if dbscan find zero cluster, return the original gaze trace
    if len(set(labels)) == 1:
        if verbose: print("DBSCAN found 0 clusters, returning original gaze trace.")
        return gaze_trace

    # Extract the cluster centers (mean of points in each cluster)
    unique_labels = set(labels)
    centroids_dbscan = []
    for label in unique_labels:
        if label != -1:  # Exclude noise
            cluster_points = gaze_trace[labels == label]
            centroid = cluster_points.mean(axis=0)
            centroids_dbscan.append(centroid)
    if verbose: print(f"DBSCAN reduced the gaze trace from {len(gaze_trace)} to {len(centroids_dbscan)} points.")
    return np.array(centroids_dbscan)