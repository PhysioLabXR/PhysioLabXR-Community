import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def save_model_parameters(weights, bias, filepath='Sweyepe_EEG.pt'):
    """
    Save model weights and bias to a file.

    Args:
        weights (torch.Tensor): Model weights
        bias (torch.Tensor): Model bias
        filepath (str): Path to save the parameters
    """
    torch.save({
        'weights': weights,
        'bias': bias
    }, filepath)
    print(f"Model parameters saved to {filepath}")

def load_model_parameters(filepath='model_parameters.pt'):
    """
    Load model weights and bias from a file.

    Args:
        filepath (str): Path to the saved parameters

    Returns:
        tuple: (weights, bias)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No model parameters found at {filepath}")

    parameters = torch.load(filepath)
    return parameters['weights'], parameters['bias']

def predict_sliding_windows(eeg_df, weights=None, bias=None, model_path='SweyepeEEGModel.pt', sampling_rate=256):
    """
    Predict EEG data using sliding windows and return timestamps of predictions.

    Args:
        eeg_df (pd.DataFrame): EEG data with 'Timestamp' column and channel columns
        weights (torch.Tensor, optional): Trained model weights. If None, loads from file
        bias (torch.Tensor, optional): Trained model bias. If None, loads from file
        model_path (str): Path to saved model parameters if weights/bias not provided
        sampling_rate (int): Sampling rate in Hz (default: 256)

    Returns:
        tuple: (timestamps_true, timestamps_false, remaining_timestamps)
            - timestamps where model predicted true
            - timestamps where model predicted false
            - timestamps in the final window that couldn't be fully processed
    """
    # Load model parameters if not provided
    if weights is None or bias is None:
        try:
            weights, bias = load_model_parameters(model_path)
            print(f"Loaded model parameters from {model_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError("Model parameters must either be provided directly or available in a saved file")

    # Constants for window sizes (matching training)
    window_size = int(1.0 * sampling_rate)  # 1 second total (0.5s pre + 0.5s post)
    step_size = int(0.1 * sampling_rate)    # 100ms steps for predictions

    # Get channel data and timestamps
    channel_data = eeg_df.filter(like='Channel').values
    timestamps = eeg_df['Timestamp'].values

    # Lists to store results
    true_predictions = []
    false_predictions = []

    # Slide window through the data
    for i in range(0, len(channel_data) - window_size, step_size):
        # Extract window
        window = channel_data[i:i+window_size]

        # Skip if window is not complete
        if len(window) < window_size:
            continue

        # Reshape to match training data format
        window_reshaped = window.reshape(1, window_size, -1)

        # Z-score normalize the window
        window_flat = window_reshaped.reshape(window_reshaped.shape[0], -1)
        window_mean = np.mean(window_flat, axis=1, keepdims=True)
        window_std = np.std(window_flat, axis=1, keepdims=True)
        window_normalized = ((window_flat - window_mean) / window_std).reshape(window_reshaped.shape)

        # Extract features (matching training feature extraction)
        features = extract_temporal_features(window_normalized)

        # Get prediction
        pred = predict_target(window_normalized, weights, bias)

        # Store timestamp (middle of window) based on prediction
        window_center_idx = i + window_size // 2
        if pred[0, 1] > 0.5:  # Check prediction for positive class
            true_predictions.append(timestamps[window_center_idx])
        else:
            false_predictions.append(timestamps[window_center_idx])

    # Get remaining timestamps that couldn't be fully processed
    last_processed_idx = len(channel_data) - window_size
    remaining_timestamps = timestamps[last_processed_idx:]

    return (
        np.array(true_predictions),
        np.array(false_predictions),
        remaining_timestamps
    )

# Example usage:
"""
# When predicting:
# Option 1: Load automatically
true_timestamps, false_timestamps, remaining = predict_sliding_windows(
    eeg_df,
    model_path='my_model.pt'  # Will load parameters from this file
)

# Option 2: Load manually and pass in
weights, bias = load_model_parameters('my_model.pt')
true_timestamps, false_timestamps, remaining = predict_sliding_windows(
    eeg_df,
    weights=weights,
    bias=bias
)
"""

"""
Helper Functions
"""


def z_norm_by_trial(X):
    """Normalize EEG data trial-wise"""
    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_normed = scaler.fit_transform(X_flat)
    return X_normed.reshape(X.shape), scaler

def extract_temporal_features(X):
    """Extract relevant ERP components"""
    vep_window = slice(0, 25)    # 0-100ms
    n200_window = slice(25, 51)  # 100-200ms
    p300_window = slice(51, 102) # 200-400ms

    features = []
    for trial in X:
        vep_features = np.max(np.abs(trial[vep_window]), axis=0)
        n200_features = np.min(trial[n200_window], axis=0)
        p300_features = np.max(trial[p300_window], axis=0)

        trial_features = np.concatenate([
            vep_features,
            n200_features,
            p300_features
        ])
        features.append(trial_features)

    return np.array(features)

def ridge_regression_gd(X, Y, X_test, Y_test, lamb=0.1, learning_rate=0.5, iterations=5000):
    """Ridge regression for target detection"""
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    n_features = X.shape[1]
    n_output = Y.shape[1]
    weights = torch.randn((n_features, n_output), requires_grad=True)
    bias = torch.randn(n_output, requires_grad=True)

    optimizer = torch.optim.Adam([weights, bias], lr=learning_rate)
    loss_func = torch.nn.BCEWithLogitsLoss()

    for k in range(iterations):
        optimizer.zero_grad()
        y_pred = torch.matmul(X, weights) + bias
        loss = loss_func(y_pred, Y)
        loss += lamb * (torch.norm(weights) + torch.norm(bias))

        loss.backward()
        optimizer.step()

        if k % 100 == 0:
            with torch.no_grad():
                test_pred = torch.matmul(X_test, weights) + bias
                test_acc = ((test_pred > 0) == Y_test).float().mean()
                print(f"Iteration {k}, Test Accuracy: {test_acc:.3f}")

    return weights.detach(), bias.detach()


def predict_target(eeg_data, weights, bias):
    """Predict if input contains target response"""
    features = extract_temporal_features(eeg_data)
    features = torch.tensor(features, dtype=torch.float32)
    logits = torch.matmul(features, weights) + bias
    probs = torch.sigmoid(logits)
    return probs.numpy()