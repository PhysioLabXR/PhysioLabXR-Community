"""Tests for RealTimeEegPreprocessor's artifact handling.

These pin the properties the WM-load calibration depends on:
- ICA must be fit at the physical-channel rank (Cyton+Daisy has no Fz/Cz/Pz;
  they are synthesized as lateral averages, so only 12 of the 15 REVE channels
  are independent).
- The cleaning function must be FROZEN after its baseline fit — a cleaning
  function that changes over time becomes a load-correlated confound.
- Component rejection must not delete frontal-midline theta (the WM signal);
  with an ocular reference supplied, blinks are removed and theta survives.
- The frontal regression stage is off by default (its trigger rate scales
  with load, biasing the load contrast).

Run: venv\\Scripts\\python -m pytest physiolabxr/examples/OpenBCIExamples/test_realtime_preprocessor.py
"""

import numpy as np
import pytest

from OpenBCIToLSLInterface import REVE_CHANNEL_NAMES, RealTimeEegPreprocessor

DAISY_LABELS = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
    "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
]
FS = 125.0
N_BOARD_ROWS = 18  # counter + 16 EEG + 1 junk row
EEG_INDICES = tuple(range(1, 17))

REVE_IDX = {name: i for i, name in enumerate(REVE_CHANNEL_NAMES)}
RAW_IDX = {name: i for i, name in enumerate(DAISY_LABELS)}

FIT_SEC = 20.0


def synth_recording(duration_sec, seed=7):
    """16-channel raw EEG: pink-ish noise + 6 Hz theta on F3/F4 + blinks on Fp1/Fp2.

    Returns (frames, blink_envelope): frames is (n, N_BOARD_ROWS) board-format,
    blink_envelope is the per-sample ocular reference (1 during a blink).
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * FS)
    t = np.arange(n) / FS

    eeg = rng.normal(0.0, 5.0, size=(n, 16))
    theta = 8.0 * np.sin(2 * np.pi * 6.0 * t)
    eeg[:, RAW_IDX["F3"]] += theta
    eeg[:, RAW_IDX["F4"]] += theta
    eeg[:, RAW_IDX["O1"]] += 6.0 * np.sin(2 * np.pi * 10.0 * t)
    eeg[:, RAW_IDX["O2"]] += 6.0 * np.sin(2 * np.pi * 10.0 * t + 0.5)

    blink_env = np.zeros(n)
    blink_len = int(0.3 * FS)
    pulse = np.hanning(blink_len)
    start = int(0.8 * FS)
    while start + blink_len < n:
        blink_env[start:start + blink_len] = np.maximum(
            blink_env[start:start + blink_len], pulse)
        start += int((1.8 + rng.uniform(0.0, 0.8)) * FS)

    eeg[:, RAW_IDX["Fp1"]] += 120.0 * blink_env
    eeg[:, RAW_IDX["Fp2"]] += 115.0 * blink_env
    eeg[:, RAW_IDX["F7"]] += 40.0 * blink_env
    eeg[:, RAW_IDX["F8"]] += 38.0 * blink_env
    eeg[:, RAW_IDX["F3"]] += 25.0 * blink_env
    eeg[:, RAW_IDX["F4"]] += 24.0 * blink_env

    frames = np.zeros((n, N_BOARD_ROWS), dtype=np.float32)
    frames[:, 0] = np.arange(n) % 256
    frames[:, 1:17] = eeg
    return frames, blink_env


def new_preprocessor(**overrides):
    kwargs = dict(
        sampling_rate=FS,
        raw_eeg_labels=DAISY_LABELS,
        ica_fit_sec=FIT_SEC,
        ica_history_sec=FIT_SEC,
    )
    kwargs.update(overrides)
    return RealTimeEegPreprocessor(**kwargs)


def run_through(preproc, frames, blink_env, chunk=25):
    outputs = []
    fit_marks = []
    for start in range(0, frames.shape[0], chunk):
        block = frames[start:start + chunk]
        ocular = blink_env[start:start + chunk]
        cleaned = preproc.process_chunk(block, EEG_INDICES, ocular_reference=ocular)
        outputs.append(cleaned)
        fit_marks.append(preproc.last_ica_fit_sample)
    return np.concatenate(outputs, axis=0), fit_marks


def band_power(signal, low, high):
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(signal.shape[0], d=1.0 / FS)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(spectrum[mask]))


def test_ica_fits_at_physical_rank():
    frames, blink = synth_recording(2 * FIT_SEC)
    preproc = new_preprocessor()
    run_through(preproc, frames, blink)

    assert preproc.ica_model is not None, "ICA must fit once enough data arrived"
    n_components = preproc.ica_model.components_.shape[0]
    assert n_components <= 12, (
        f"Cyton+Daisy has 12 physical REVE channels (Fz/Cz/Pz are synthesized); "
        f"fitting {n_components} components is rank-deficient")


def test_ica_frozen_after_first_fit():
    frames, blink = synth_recording(4 * FIT_SEC)
    preproc = new_preprocessor()
    _, fit_marks = run_through(preproc, frames, blink)

    marks = [m for m in fit_marks if m is not None]
    assert marks, "ICA never fit"
    assert len(set(marks)) == 1, (
        "the cleaning function must be frozen after the baseline fit — refitting "
        "mid-session makes cleaning differ between load blocks")


def test_blink_removed_and_theta_preserved_with_ocular_reference():
    frames, blink = synth_recording(3 * FIT_SEC)
    preproc = new_preprocessor()
    cleaned, _ = run_through(preproc, frames, blink)

    n_eval = int(FIT_SEC * FS)
    eval_cleaned = cleaned[-n_eval:]
    eval_blink = blink[-n_eval:] > 0.5

    raw_eeg = frames[-n_eval:, 1:17].astype(float)

    # Blink attenuation at Fp1: compare within-blink variance, cleaned vs raw.
    raw_fp1 = raw_eeg[:, RAW_IDX["Fp1"]] - np.median(raw_eeg[:, RAW_IDX["Fp1"]])
    cleaned_fp1 = eval_cleaned[:, REVE_IDX["Fp1"]] - np.median(eval_cleaned[:, REVE_IDX["Fp1"]])
    assert eval_blink.sum() > 50, "evaluation span must contain blinks"
    raw_blink_rms = float(np.sqrt(np.mean(raw_fp1[eval_blink] ** 2)))
    cleaned_blink_rms = float(np.sqrt(np.mean(cleaned_fp1[eval_blink] ** 2)))
    assert cleaned_blink_rms < 0.6 * raw_blink_rms, (
        f"blinks must be attenuated: cleaned RMS {cleaned_blink_rms:.1f} vs raw {raw_blink_rms:.1f}")

    # Theta preservation at F3: cleaned 5-7 Hz power stays near the raw level.
    raw_theta = band_power(raw_eeg[:, RAW_IDX["F3"]], 5.0, 7.0)
    cleaned_theta = band_power(eval_cleaned[:, REVE_IDX["F3"]], 5.0, 7.0)
    assert cleaned_theta > 0.6 * raw_theta, (
        f"frontal theta is the WM signal and must survive cleaning: "
        f"cleaned {cleaned_theta:.1f} vs raw {raw_theta:.1f}")


def test_theta_preserved_without_ocular_reference_too():
    frames, blink = synth_recording(3 * FIT_SEC)
    preproc = new_preprocessor()

    outputs = []
    for start in range(0, frames.shape[0], 25):
        outputs.append(preproc.process_chunk(frames[start:start + 25], EEG_INDICES))
    cleaned = np.concatenate(outputs, axis=0)

    n_eval = int(FIT_SEC * FS)
    raw_theta = band_power(frames[-n_eval:, 1 + RAW_IDX["F3"]].astype(float), 5.0, 7.0)
    cleaned_theta = band_power(cleaned[-n_eval:, REVE_IDX["F3"]], 5.0, 7.0)
    assert cleaned_theta > 0.6 * raw_theta, (
        "without ocular evidence the rejection must stay conservative — "
        "never delete frontal theta on spectral/spatial heuristics alone")


def test_synthesized_midline_is_average_of_cleaned_laterals():
    frames, blink = synth_recording(2 * FIT_SEC)
    preproc = new_preprocessor()
    cleaned, _ = run_through(preproc, frames, blink)

    tail = cleaned[-int(5 * FS):]
    expected_fz = 0.5 * (tail[:, REVE_IDX["F3"]] + tail[:, REVE_IDX["F4"]])
    np.testing.assert_allclose(
        tail[:, REVE_IDX["Fz"]], expected_fz, atol=1e-3,
        err_msg="Fz must be synthesized from the CLEANED F3/F4 (midline channels "
                "carry no independent signal and must not pass through ICA)")


def test_frontal_regression_disabled_by_default():
    preproc = new_preprocessor()
    assert preproc.frontal_regression_strength == 0.0, (
        "the frontal regression stage triggers more often under high load "
        "(more eye movement) and must be opt-in, not default")


def test_default_fit_window_is_long_and_refit_disabled():
    preproc = RealTimeEegPreprocessor(sampling_rate=FS, raw_eeg_labels=DAISY_LABELS)
    assert preproc.fit_ica_sample_count >= int(60 * FS), (
        "the default ICA fit window must be at least 60 s (opening rest + practice)")
    assert preproc.ica_refit_disabled, (
        "periodic refits must be off by default (frozen cleaning function)")


# --- Gaze ocular-reference helpers (feed the live ICA validation) -------------

from OpenBCIToLSLInterface import compute_ocular_value, resample_ocular_reference


def test_ocular_value_blink_dominates():
    fwd = np.array([0.0, 0.0, 1.0])
    value = compute_ocular_value(fwd, fwd, dt=0.005, combined_valid=0.0,
                                 left_valid=0.0, right_valid=0.0)
    assert value == pytest.approx(1.0), "eye-invalid samples (blinks) must saturate the trace"


def test_ocular_value_scales_with_gaze_speed():
    prev = np.array([0.0, 0.0, 1.0])
    angle = 0.02  # rad over 5 ms -> 4 rad/s = full scale
    curr = np.array([np.sin(angle), 0.0, np.cos(angle)])
    value = compute_ocular_value(prev, curr, dt=0.005, combined_valid=1.0,
                                 left_valid=1.0, right_valid=1.0)
    assert value == pytest.approx(1.0, abs=0.05)

    slow = np.array([np.sin(angle / 10), 0.0, np.cos(angle / 10)])
    value_slow = compute_ocular_value(prev, slow, dt=0.005, combined_valid=1.0,
                                      left_valid=1.0, right_valid=1.0)
    assert 0.05 < value_slow < 0.2, "fixational drift must stay near zero"


def test_resample_reference_sample_holds_latest_gaze_value():
    buffer = [(0.0, 0.0), (0.5, 1.0)]
    values = resample_ocular_reference(buffer, n_samples=4, sampling_rate=4.0, t_end=1.0)
    np.testing.assert_allclose(values, [0.0, 1.0, 1.0, 1.0])


def test_resample_reference_empty_buffer_returns_none():
    assert resample_ocular_reference([], n_samples=8, sampling_rate=125.0, t_end=10.0) is None


# --- Montage handling ---------------------------------------------------------
# The BrainFlow default Cyton+Daisy labels carry no Fz/Cz/Pz, so those REVE
# channels are synthesized. If the actual cap places electrodes on the midline,
# declaring the true montage must make the pipeline use the real recordings.

MIDLINE_LABELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "C3",
    "Cz", "C4", "P3", "Pz", "P4", "O1", "O2", "T7",
]


def test_true_midline_montage_uses_all_fifteen_physical_channels():
    preproc = RealTimeEegPreprocessor(
        sampling_rate=FS, raw_eeg_labels=MIDLINE_LABELS)
    assert len(preproc.physical_positions) == 15
    assert preproc.synthesized_outputs == [], (
        "with real Fz/Cz/Pz electrodes nothing may be synthesized — "
        "ICA then runs on all 15 physical channels")


def test_interface_accepts_montage_override():
    from OpenBCIToLSLInterface import OpenBCIToLSLInterface
    iface = OpenBCIToLSLInterface(
        stream_name="montage_test",
        serial_port="COM99",
        board_id="cyton+daisy",
        eeg_labels_override=MIDLINE_LABELS,
    )
    assert iface.raw_eeg_labels == MIDLINE_LABELS
    assert iface.eeg_preprocessor.synthesized_outputs == []


def test_interface_rejects_wrong_override_length():
    from OpenBCIToLSLInterface import OpenBCIToLSLInterface
    with pytest.raises(ValueError):
        OpenBCIToLSLInterface(
            stream_name="montage_test",
            serial_port="COM99",
            board_id="cyton+daisy",
            eeg_labels_override=["Fp1", "Fp2"],
        )
