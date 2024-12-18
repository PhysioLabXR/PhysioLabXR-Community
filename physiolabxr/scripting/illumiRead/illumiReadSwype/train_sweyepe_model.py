import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def compute_erp(X, conditions, fs, condition_label=1):
    """
    Compute the ERP for a given condition.

    Args:
        X (np.ndarray): EEG data of shape (N, channels, samples)
        conditions (np.ndarray): Array of shape (N,) with condition labels (0 or 1)
        fs (int): Sampling rate
        condition_label (int): Condition for which to compute ERP (e.g., 0 or 1)

    Returns:
        mean_erp (np.ndarray): shape (channels, samples)
        sem_erp (np.ndarray): shape (channels, samples)
        time (np.ndarray): time vector in seconds for plotting
    """
    # Select trials for the given condition
    cond_trials = X[conditions == condition_label]

    # Compute mean and SEM
    mean_erp = np.mean(cond_trials, axis=0)  # (channels, samples)
    sem_erp = np.std(cond_trials, axis=0, ddof=1) / np.sqrt(cond_trials.shape[0])

    # Create time vector in seconds
    samples = X.shape[2]
    time = np.arange(samples) / fs - 0.5  # assuming 0s is event onset

    return mean_erp, sem_erp, time


def z_norm_by_trial(X):
    """Normalize EEG data trial-wise"""
    scaler = StandardScaler()
    X_flat = X.reshape(X.shape[0], -1)
    X_normed = scaler.fit_transform(X_flat)
    return X_normed.reshape(X.shape), scaler


def extract_temporal_features(X):
    """Extract relevant ERP components"""
    vep_window = slice(0, 25)  # 0-100ms
    n200_window = slice(25, 51)  # 100-200ms
    p300_window = slice(51, 102)  # 200-400ms

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


def extract_epochs_from_combined(data, window_size=256):
    """Extract epochs from combined data format using Mark column

    Args:
        data: DataFrame with Timestamp, Channel_*, and Mark columns
        window_size: Number of samples to include in each epoch (default 256 = 1s at 256Hz)
    """
    # Find all non-zero marks which indicate events
    event_indices = data.index[data['Mark'] != 0].tolist()

    epochs = []
    conditions = []

    half_window = window_size // 2

    for idx in event_indices:
        if idx >= half_window and idx + half_window < len(data):
            # Extract window around event
            window_data = data.iloc[idx - half_window:idx + half_window]

            # Get channel data only
            channel_cols = [col for col in data.columns if col.startswith('Channel')]
            epoch_data = window_data[channel_cols].values

            # Get condition from Mark column
            condition = data.iloc[idx]['Mark']

            epochs.append(epoch_data)
            conditions.append(condition)

    return np.array(epochs), np.array(conditions)


def save_model_parameters(weights, bias, filepath='SweyepeEEGModel'):
    """Save model weights and bias to a file."""
    torch.save({
        'weights': weights,
        'bias': bias
    }, filepath)
    print(f"Model parameters saved to {filepath}")


def train_model(data_path):
    # Load combined data
    data = pd.read_csv(data_path)
    print(f"Loaded data with shape: {data.shape}")

    # Extract epochs
    X, conditions = extract_epochs_from_combined(data)
    print(f"Extracted {len(X)} epochs")

    # Filter for valid conditions (assuming 1 & 2 are valid marks)
    mask = np.isin(conditions, [1, 2])
    X = X[mask]
    conditions = conditions[mask]
    conditions = conditions - 1  # Adjust to 0/1
    n_conditions = 2

    # Balance classes
    wrong_indices = np.where(conditions == 0)[0]
    correct_indices = np.where(conditions == 1)[0]

    np.random.seed(42)
    if len(wrong_indices) > len(correct_indices):
        wrong_sampled = np.random.choice(wrong_indices, size=len(correct_indices), replace=False)
        balanced_indices = np.sort(np.concatenate([wrong_sampled, correct_indices]))
    else:
        correct_sampled = np.random.choice(correct_indices, size=len(wrong_indices), replace=False)
        balanced_indices = np.sort(np.concatenate([wrong_indices, correct_sampled]))

    X = X[balanced_indices]
    conditions = conditions[balanced_indices]

    # Create design matrix
    dm = np.zeros((len(conditions), n_conditions))
    for i, condition in enumerate(conditions):
        dm[i, int(condition)] = 1

    # Split data
    X_train, X_test, dm_train, dm_test = train_test_split(
        X, dm, test_size=0.1, stratify=conditions, random_state=42
    )

    # Normalize data
    X_train_norm, _ = z_norm_by_trial(X_train)
    X_test_norm, _ = z_norm_by_trial(X_test)

    # Extract features and train
    X_train_features = extract_temporal_features(X_train_norm)
    X_test_features = extract_temporal_features(X_test_norm)

    weights, bias = ridge_regression_gd(
        X=X_train_features,
        Y=dm_train,
        X_test=X_test_features,
        Y_test=dm_test
    )

    # Save the model
    save_model_parameters(weights, bias)

    # After training, also plot ERPs for the entire balanced dataset
    # Normalize the entire balanced dataset for ERP plotting
    X_norm, _ = z_norm_by_trial(X)

    fs = 300  # Sampling rate (adjust if different)
    mean_target, sem_target, time = compute_erp(X_norm, conditions, fs, condition_label=1)
    mean_distractor, sem_distractor, _ = compute_erp(X_norm, conditions, fs, condition_label=0)

    num_channels = X_norm.shape[1]
    rows = 6
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12), sharex=True, sharey=True)
    axes = axes.flatten()

    for ch in range(num_channels):
        ax = axes[ch]
        # Plot distractor condition
        ax.plot(time, mean_distractor[ch, :], color='blue', label='Distractor' if ch == 0 else None)
        ax.fill_between(time,
                        mean_distractor[ch, :] - sem_distractor[ch, :],
                        mean_distractor[ch, :] + sem_distractor[ch, :],
                        color='blue', alpha=0.3)

        # Plot target condition
        ax.plot(time, mean_target[ch, :], color='red', label='Target' if ch == 0 else None)
        ax.fill_between(time,
                        mean_target[ch, :] - sem_target[ch, :],
                        mean_target[ch, :] + sem_target[ch, :],
                        color='red', alpha=0.3)

        ax.axvline(0, color='k', linestyle='--')
        ax.set_title(f"Channel {ch}")
        if ch % cols == 0:
            ax.set_ylabel('Amplitude (a.u.)')

    # Add a single legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')

    fig.text(0.5, 0.04, 'Time (s)', ha='center')
    fig.suptitle("ERPs for All Channels - Target vs Distractor", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

    return weights, bias


