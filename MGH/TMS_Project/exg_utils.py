import numpy as np

def my_range(start, num_of_elements, interval):
    return np.array([start + interval * n for n in range(num_of_elements)])


# -2 -1 0 1 2
def collect_evoked_responses(data, data_ts, event_markers, event_markers_ts, samples_before=100, samples_after = 500, ts_shift=True, event_marker_axis=1, event_id=1):
    evoked_responses = []
    event_index = np.where(event_markers == event_id)[event_marker_axis]
    event_timestamps = event_markers_ts[event_index]
    # np.where(data[DataStreamName][1] > time_stamp)[0][0]
    # data_event_timestamp = []
    for event_timestamp in event_timestamps:
        data_event_timestamp = np.where(data_ts >= event_timestamp)[0][0]
        evoked_response = data[:, data_event_timestamp-samples_before:data_event_timestamp+samples_after]
        evoked_ts = data_ts[data_event_timestamp-samples_before:data_event_timestamp+samples_after]
        if ts_shift:
            evoked_ts = evoked_ts-data_ts[data_event_timestamp]
        evoked_responses.append({
            'evoked_response':evoked_response,
            'evoked_ts':evoked_ts
        })


        # data_event_timestamp.append(np.where(data_ts > event_timestamp)[0][0])
        # data

    # time_stamps = event_timestamps.time
    return evoked_responses
