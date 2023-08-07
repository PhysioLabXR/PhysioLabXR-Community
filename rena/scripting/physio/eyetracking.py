import numpy as np

from rena.scripting.physio.utils import interpolate_array_nan


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