import numpy as np

from physiolabxr.scripting.physio.utils import interpolate_array_nan, time_to_index


def gap_fill(gaze_xyz, gaze_status, valid_status, gaze_timestamps, max_gap_time=0.075, verbose=True):
    valid_diff = np.diff(np.concatenate([[valid_status], gaze_status, [valid_status]]))
    gap_start_indices = np.where(valid_diff < 0)[0]
    gap_end_indices = np.where(valid_diff > 0)[0]
    gaze_timestamps_extended = np.append(gaze_timestamps, gaze_timestamps[-1])
    ignored_gap_start_end_indices = []

    interpolated_gap_durations = []
    ignored_gap_durations = []
    interpolated_gap_count = 0

    for start, end in zip(gap_start_indices, gap_end_indices):
        if (gap_duration := gaze_timestamps_extended[end] - gaze_timestamps_extended[start]) > max_gap_time:
            ignored_gap_durations.append(gap_duration)
            ignored_gap_start_end_indices.append((start, end))
            continue
        else:
            interpolated_gap_count += 1
            gaze_xyz[:, start: end] = np.nan  # change the gaps to be interpolated to nan
            interpolated_gap_durations.append(gap_duration)
    # plt.hist(interpolated_gap_durations + ignored_gap_durations, bins=100)
    # plt.show()
    if verbose: print(f"With max gap duration {max_gap_time * 1e3}ms, \n {interpolated_gap_count} gaps are interpolated among {len(gap_start_indices)} gaps, \n with interpolated gap with mean:median duration {np.mean(interpolated_gap_durations) *1e3}ms:{np.median(interpolated_gap_durations) *1e3}ms, \n and ignored gap with mean:median duration {np.mean(ignored_gap_durations) *1e3}ms:{np.median(ignored_gap_durations) *1e3}ms ")
    # interpolate the gaps
    gaze_xyz = interpolate_array_nan(gaze_xyz)

    # change the ignored gaps to nan
    for start, end in ignored_gap_start_end_indices:  # now change the gap that are ignored to be nan and they remain so at return
        gaze_xyz[:, start: end] = np.nan

    return gaze_xyz

def _calculate_gaze_angles(gaze_vector):
    """
    gaze vectors should be 3D vectors in the eye coordinate system, with the z axis pointing out of the eye straight ahead
    @param gaze_vector:
    @param head_rotation_xy_degree:
    @return:
    """
    reference_vector = np.array([0, 0, 1])
    dot_products = np.dot(gaze_vector.T, reference_vector)
    magnitudes = np.linalg.norm(gaze_vector, axis=0)
    reference_magnitude = np.linalg.norm(reference_vector)
    cosine_angles = dot_products / (magnitudes * reference_magnitude)
    angles_rad = np.arccos(cosine_angles)
    angles_deg = np.degrees(angles_rad)

    return angles_deg


def _compute_dispersion(angles):
    """
    computes dispersion within an index window
    @param gaze_angles_degree:
    @param gaze_point_window_start_index:
    @param gaze_point_window_end_index:
    @return:
    """
    return np.std(angles)


def fixation_detection_idt(gaze_xyz, timestamps, window_size=0.175, dispersion_threshold_degree=0.5, saccade_min_sample=2, return_last_window_start=False):
    """

    @param gaze_xyz:
    @param timestamps:
    @param window_size:
    @param dispersion_threshold_degree:
    @param saccade_min_sample: the minimal number of samples between consecutive fixations to be considered as a saccade
    @return:
    """
    assert window_size > 0, "fixation_detection_idt: window size must be positive"
    gaze_angles_degree = _calculate_gaze_angles(gaze_xyz)
    windows = [(i, time_to_index(timestamps, t + window_size)) for i, t in enumerate(timestamps)]
    last_window_start = 0
    fixations = []
    for start, end in windows:
        if end >= len(timestamps):
            break
        if end - start < saccade_min_sample:
            continue
        center_time = timestamps[start] + window_size / 2
        if _compute_dispersion(gaze_angles_degree[start:end]) < dispersion_threshold_degree:
            fixations.append([1, center_time])  # 1 for fixation
        else:
            fixations.append([0, center_time])
        last_window_start = start
    if return_last_window_start:
        return np.array(fixations).T, last_window_start
    else:
        return np.array(fixations).T