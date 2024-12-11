import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import zscore
from scipy.interpolate import interp1d

from physiolabxr.scripting.illumiRead.illumiReadSwype.eeg_model import predict_sliding_windows
from physiolabxr.scripting.illumiRead.illumiReadSwype.eeg_preprocessing import preprocess_eeg


def full_eeg_pipeline(eeg_data, timestamps, model_path=r'C:\Users\6173-group\Documents\PhysioLabXR\physiolabxr\scripting\illumiRead\illumiReadSwype\SweyepeEEGModel.pt'):
    """
    Complete EEG pipeline: preprocessing + prediction
    """
    # First preprocess the EEG
    preprocessed = preprocess_eeg(eeg_data, timestamps)


    # Convert to DataFrame for prediction
    data_dict = {'Timestamp': preprocessed['timestamps']}
    for i in range(preprocessed['processed_data'].shape[0]):
        channel_name = f'Channel_{i + 1}'
        if i in preprocessed['bad_channels']:
            channel_name += '_interpolated'
        data_dict[channel_name] = preprocessed['processed_data'][i, :]

    eeg_df = pd.DataFrame(data_dict)

    # Get predictions
    true_timestamps, false_timestamps, remaining = predict_sliding_windows(
        eeg_df,
        model_path=model_path,
        sampling_rate=int(preprocessed['sampling_rate'])
    )

    return {
        'preprocessed': preprocessed,
        'predictions': {
            'true_timestamps': true_timestamps,
            'false_timestamps': false_timestamps,
            'remaining': remaining
        }
    }

# Run the pipeline:
# results = full_eeg_pipeline(eeg_data, timestamps)
# Print predictions:
# print("Detected events at:", results['predictions']['true_timestamps'])