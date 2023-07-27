import copy

import numpy as np


def get_event_locked_data(event_marker, data, events_of_interest, tmin, tmax, srate, return_last_event_time=False, verbose=None):
    """
    @param event_marker: tuple of event marker and its timestamps
    @param data: tuple of data and its timestamps
    @param events_of_interest: iterable of event markers with which to get event aligned data
    @param return_last_event_time: whether to return the time of the last found in the data

    @return: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    """
    assert tmin < tmax, 'tmin must be less than tmax'
    event_marker, event_marker_time = event_marker
    event_marker = event_marker[0]
    data, data_time = data
    events_of_interest = [e for e in events_of_interest if e in event_marker]
    rtn = {e: [] for e in events_of_interest}
    latest_event_start_time = -1
    epoch_length = int((tmax - tmin) * srate)
    for e in events_of_interest:
        this_event_marker_time = event_marker_time[event_marker == e]
        data_event_starts = [np.argmin(abs(data_time - (s+tmin))) for s in this_event_marker_time]
        data_event_ends = [epoch_length + s for s in data_event_starts]
        for i, j, e_time in zip(data_event_starts, data_event_ends, this_event_marker_time):
            if j < len(data_time):
                rtn[e].append(data[:, i:j])
                latest_event_start_time = max(latest_event_start_time, e_time)
    # convert to numpy arrays
    rtn = {k: np.array(v) for k, v in rtn.items() if len(v) > 0}
    if verbose:
        [print(f"Found {len(v)} events for event marker {k}") for k, v in rtn.items()]
    if return_last_event_time:
        return rtn, latest_event_start_time
    else:
        return rtn


def buffer_event_locked_data(event_locked_data, buffer: dict):
    """
    @param event_locked_data: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    @param buffer: dictionary of event marker and its corresponding buffer. The keys are the event markers
    @return: dictionary of event marker and its corresponding event locked data. The keys are the event markers
    """
    rtn = copy.deepcopy(buffer)
    for k, v in event_locked_data.items():
        if k in buffer:
            v = np.concatenate([buffer[k], v], axis=0)
        else:
            v = np.array(v)
        rtn[k] = v
    return rtn


def get_baselined_event_locked_data(event_locked_data, pick: int, baseline_t, srate):
    rtn = {}
    pick = [pick] if isinstance(pick, int) else pick
    for k, v in event_locked_data.items():
        d = v[:, pick]
        d = d - np.mean(d[:, :, :int(baseline_t * srate)], axis=2, keepdims=True)
        rtn[k] = d
    return rtn