from collections import deque
import warnings

import brainflow
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamInlet, StreamOutlet, local_clock, resolve_byprop
import serial
from serial.tools import list_ports

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, sosfilt, sosfilt_zi
from scipy.stats import kurtosis

try:
    from sklearn.decomposition import FastICA
    from sklearn.exceptions import ConvergenceWarning
except ImportError:
    FastICA = None
    ConvergenceWarning = Warning


REVE_CHANNEL_NAMES = (
    "Fp1", "Fp2", "F7", "F3", "Fz",
    "F4", "F8", "C3", "Cz", "C4",
    "P3", "Pz", "P4", "O1", "O2",
)

# The lab cap's ACTUAL wiring, board pins 1-16 (Cyton 1-8, Daisy 9-16). This is
# lab-authored ground truth — it must match the raw-stream preset
# physiolabxr/_presets/LSLPresets/OpenBCICytonDaisy15.json (ChannelNames[1:17]),
# and a CLAdEEG test enforces that. It includes REAL Fz/Cz/Pz electrodes, unlike
# BrainFlow's default labels — passing it as eeg_labels_override makes the
# preprocessing use the true positions instead of synthesizing the midline.
CAP_MONTAGE = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "O2",
    "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "Unused",
)

MIDLINE_SOURCE_MAP = {
    "Fz": ("F3", "F4"),
    "Cz": ("C3", "C4"),
    "Pz": ("P3", "P4"),
}

FRONTAL_CHANNELS = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}
POSTERIOR_CHANNELS = {"P3", "Pz", "P4", "O1", "O2"}


def normalize_channel_label(value):
    text = str(value or "").strip().upper()
    return "".join(character for character in text if character.isalnum())


def common_average_reference(samples):
    if samples.ndim != 2 or samples.shape[0] < 2:
        return samples
    return samples - np.mean(samples, axis=0, keepdims=True)


def robust_clip_channels(samples, n_sigmas=6.0):
    clipped = np.asarray(samples, dtype=float).copy()
    if clipped.ndim != 2 or clipped.shape[-1] == 0:
        return clipped.astype(np.float32, copy=False)

    for idx in range(clipped.shape[0]):
        channel = clipped[idx]
        finite_values = channel[np.isfinite(channel)]
        if finite_values.size < 8:
            continue

        median = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - median)))
        sigma = 1.4826 * mad
        if sigma <= 0:
            continue

        limit = n_sigmas * sigma
        clipped[idx] = np.clip(channel, median - limit, median + limit)

    return clipped.astype(np.float32, copy=False)


def compute_ocular_value(previous_forward, forward, dt, combined_valid, left_valid, right_valid,
                         full_scale_rad_per_s=4.0):
    """One ocular-activity value in [0, 1] from a Varjo gaze sample: validity
    dropouts (blinks) saturate the trace; otherwise gaze angular speed scales it
    (4 rad/s ~ 230 deg/s, comfortably inside saccade range, maps to 1.0)."""
    invalidity = 1.0 - min(float(combined_valid), float(left_valid), float(right_valid))
    if invalidity >= 1.0:
        return 1.0

    speed_term = 0.0
    if previous_forward is not None and dt > 0:
        prev = np.asarray(previous_forward, dtype=float)
        curr = np.asarray(forward, dtype=float)
        prev_norm = np.linalg.norm(prev)
        curr_norm = np.linalg.norm(curr)
        if prev_norm > 1e-9 and curr_norm > 1e-9:
            cosine = float(np.clip(np.dot(prev, curr) / (prev_norm * curr_norm), -1.0, 1.0))
            angle = float(np.arccos(cosine))
            speed_term = min(1.0, (angle / dt) / float(full_scale_rad_per_s))

    return max(invalidity, speed_term)


def resample_ocular_reference(buffer, n_samples, sampling_rate, t_end):
    """Sample-holds a [(timestamp, value)] gaze buffer onto the n_samples EEG grid
    ending at t_end. Returns None when no gaze data is available."""
    if not buffer or n_samples <= 0 or sampling_rate <= 0:
        return None

    values = np.empty(n_samples, dtype=np.float32)
    buffer_idx = 0
    last_value = np.nan
    for sample_idx in range(n_samples):
        t = t_end - (n_samples - 1 - sample_idx) / float(sampling_rate)
        while buffer_idx < len(buffer) and buffer[buffer_idx][0] <= t:
            last_value = buffer[buffer_idx][1]
            buffer_idx += 1
        if not np.isfinite(last_value):
            # EEG samples preceding the first gaze sample hold the first value.
            last_value = buffer[0][1]
        values[sample_idx] = last_value
    return values


class GazeOcularReferenceInlet:
    """Non-blocking subscriber to the Varjo gaze LSL stream that maintains a
    rolling ocular-activity trace for validating ICA component rejection.
    Degrades to None (conservative preprocessing) when the stream is absent."""

    STREAM_NAME = "VarjoEyeTrackingLSL"
    COMBINED_VALID_IDX = 9
    FORWARD_IDX = (10, 11, 12)
    LEFT_VALID_IDX = 17
    RIGHT_VALID_IDX = 27

    def __init__(self, buffer_sec=180.0, resolve_retry_sec=5.0):
        self.buffer_sec = float(buffer_sec)
        self.resolve_retry_sec = float(resolve_retry_sec)
        self.buffer = deque()
        self._inlet = None
        self._next_resolve_at = 0.0
        self._previous_forward = None
        self._previous_ts = None
        self._connected_logged = False

    def _ensure_inlet(self):
        if self._inlet is not None:
            return
        now = local_clock()
        if now < self._next_resolve_at:
            return
        self._next_resolve_at = now + self.resolve_retry_sec
        try:
            candidates = resolve_byprop("name", self.STREAM_NAME, timeout=0.0)
        except Exception:
            candidates = []
        if candidates:
            try:
                self._inlet = StreamInlet(candidates[0], max_buflen=30)
                if not self._connected_logged:
                    print(f"OpenBCIInterface: connected gaze stream `{self.STREAM_NAME}` "
                          "for ocular-validated ICA.")
                    self._connected_logged = True
            except Exception as error:
                print(f"OpenBCIInterface: could not open gaze inlet: {error}")
                self._inlet = None

    def poll(self):
        self._ensure_inlet()
        if self._inlet is None:
            return
        try:
            samples, timestamps = self._inlet.pull_chunk(timeout=0.0, max_samples=512)
        except Exception:
            return
        for sample, ts in zip(samples, timestamps):
            forward = np.asarray([sample[i] for i in self.FORWARD_IDX], dtype=float)
            dt = (ts - self._previous_ts) if self._previous_ts is not None else 0.0
            value = compute_ocular_value(
                self._previous_forward, forward, dt,
                sample[self.COMBINED_VALID_IDX],
                sample[self.LEFT_VALID_IDX],
                sample[self.RIGHT_VALID_IDX],
            )
            self.buffer.append((float(ts), float(value)))
            self._previous_forward = forward
            self._previous_ts = ts

        cutoff = local_clock() - self.buffer_sec
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def reference_for(self, n_samples, sampling_rate):
        """The ocular trace aligned with the newest n_samples EEG samples, or None."""
        self.poll()
        if not self.buffer:
            return None
        return resample_ocular_reference(
            list(self.buffer), n_samples, sampling_rate, t_end=local_clock())


class RealTimeEegPreprocessor:
    """Causal notch/band-pass -> rank-aware frozen FastICA (physical channels only,
    ocular-evidence component rejection) -> midline synthesis -> CAR -> robust clip.

    Design constraints for WM-load calibration (why the defaults look like this):
    - The Cyton+Daisy montage has no physical Fz/Cz/Pz; they are synthesized as
      lateral averages, so ICA must run on the physical channels only or the
      decomposition is rank-deficient.
    - The ICA fit is FROZEN after one long baseline fit (default 90 s): a cleaning
      function that changes over time cleans different load blocks differently and
      becomes a load-correlated confound.
    - A component is only rejected as ocular when it is frontally weighted AND its
      time course correlates with concurrent ocular activity (gaze velocity /
      validity dropouts supplied per sample via `ocular_reference`). Spectral or
      spatial heuristics alone match frontal-midline theta — the WM signal itself.
      Without an ocular reference the fallback is deliberately conservative
      (extreme kurtosis only).
    - The frontal regression stage triggers more often under high load (more eye
      movement), so it is OFF by default.
    """

    def __init__(
        self,
        *,
        sampling_rate,
        raw_eeg_labels,
        low_hz=1.0,
        high_hz=40.0,
        line_noise_hz=60.0,
        enable_ica=True,
        ica_history_sec=90.0,
        ica_fit_sec=90.0,
        ica_refit_sec=None,
        ica_max_iter=400,
        ica_random_state=42,
        ica_source_z_threshold=4.0,
        ica_kurtosis_z_threshold=3.0,
        ica_frontal_ratio_threshold=1.10,
        ocular_corr_threshold=0.30,
        robust_clip_sigma=6.0,
        frontal_artifact_z_threshold=3.25,
        frontal_regression_strength=0.0,
        artifact_smoothing_sec=0.08,
        max_artifact_attenuation=0.30,
    ):
        self.sampling_rate = float(sampling_rate)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.line_noise_hz = float(line_noise_hz)
        self.enable_ica = bool(enable_ica) and FastICA is not None
        self.ica_max_iter = int(ica_max_iter)
        self.ica_random_state = int(ica_random_state)
        self.ica_source_z_threshold = float(ica_source_z_threshold)
        self.ica_kurtosis_z_threshold = float(ica_kurtosis_z_threshold)
        self.ica_frontal_ratio_threshold = float(ica_frontal_ratio_threshold)
        self.ocular_corr_threshold = float(ocular_corr_threshold)
        self.robust_clip_sigma = float(robust_clip_sigma)
        self.frontal_artifact_z_threshold = float(frontal_artifact_z_threshold)
        self.frontal_regression_strength = float(frontal_regression_strength)
        self.max_artifact_attenuation = float(max_artifact_attenuation)
        self.project_specs = self._build_project_specs(raw_eeg_labels)

        # Physical vs synthesized REVE channels. ICA sees only physical ones.
        self.physical_positions = tuple(
            idx for idx, spec in enumerate(self.project_specs) if len(spec) == 1
        )
        reve_to_physical = {reve_idx: pos for pos, reve_idx in enumerate(self.physical_positions)}
        self.synthesized_outputs = []
        for reve_idx, spec in enumerate(self.project_specs):
            if len(spec) == 1:
                continue
            source_labels = MIDLINE_SOURCE_MAP[REVE_CHANNEL_NAMES[reve_idx]]
            source_positions = []
            for label in source_labels:
                source_reve_idx = REVE_CHANNEL_NAMES.index(label)
                source_positions.append(reve_to_physical[source_reve_idx])
            self.synthesized_outputs.append((reve_idx, tuple(source_positions)))

        self.frontal_indices = tuple(
            idx for idx, label in enumerate(REVE_CHANNEL_NAMES) if label in FRONTAL_CHANNELS
        )
        self.posterior_indices = tuple(
            idx for idx, label in enumerate(REVE_CHANNEL_NAMES) if label in POSTERIOR_CHANNELS
        )
        self.physical_frontal_positions = tuple(
            pos for pos, reve_idx in enumerate(self.physical_positions)
            if REVE_CHANNEL_NAMES[reve_idx] in FRONTAL_CHANNELS
        )

        self.artifact_kernel_len = max(int(round(float(artifact_smoothing_sec) * self.sampling_rate)), 1)
        history_seconds = max(float(ica_history_sec), float(ica_fit_sec))
        max_history = max(int(round(history_seconds * self.sampling_rate)), 1)
        self.ica_history = deque(maxlen=max_history)
        self.ocular_history = deque(maxlen=max_history)
        self.fit_ica_sample_count = max(int(round(float(ica_fit_sec) * self.sampling_rate)), 256)
        self.ica_refit_disabled = ica_refit_sec is None
        self.refit_ica_sample_count = (
            None if self.ica_refit_disabled
            else max(int(round(float(ica_refit_sec) * self.sampling_rate)), 1)
        )
        self.samples_seen = 0
        self.last_ica_fit_sample = None
        self.ica_model = None
        self.ica_component_mask = None
        self.ica_source_limits = None
        self.ica_ready_logged = False
        self.ocular_fallback_warned = False

        self.bandpass_sos = self._build_bandpass_sos()
        self.bandpass_states = None
        self.notch_coeffs = self._build_notch_coeffs()
        self.notch_states = None

        if bool(enable_ica) and FastICA is None:
            print("OpenBCIInterface: scikit-learn is unavailable, ICA preprocessing is disabled.")

    @staticmethod
    def _build_project_specs(raw_eeg_labels):
        label_to_idx = {
            normalize_channel_label(label): idx
            for idx, label in enumerate(raw_eeg_labels)
        }
        specs = []
        for label in REVE_CHANNEL_NAMES:
            key = normalize_channel_label(label)
            if key in label_to_idx:
                specs.append((label_to_idx[key],))
                continue

            source_labels = MIDLINE_SOURCE_MAP.get(label)
            if source_labels is None:
                raise RuntimeError(f"Cannot map required REVE channel `{label}` from raw OpenBCI labels.")
            source_indices = []
            for source_label in source_labels:
                source_key = normalize_channel_label(source_label)
                if source_key not in label_to_idx:
                    raise RuntimeError(
                        f"Cannot synthesize `{label}` because source channel `{source_label}` is missing."
                    )
                source_indices.append(label_to_idx[source_key])
            specs.append(tuple(source_indices))
        return tuple(specs)

    def _build_bandpass_sos(self):
        nyquist = self.sampling_rate / 2.0
        high = min(self.high_hz, nyquist * 0.95)
        low = max(self.low_hz, 0.01)
        if low >= high:
            return None
        return butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")

    def _build_notch_coeffs(self):
        nyquist = self.sampling_rate / 2.0
        if self.line_noise_hz <= 0 or self.line_noise_hz >= nyquist * 0.95:
            return None
        if self.line_noise_hz >= self.high_hz:
            return None
        return iirnotch(self.line_noise_hz, 30.0, fs=self.sampling_rate)

    def _project_physical(self, board_frames, eeg_channel_indices):
        """Extracts only the physically recorded REVE channels (no synthesized midline)."""
        eeg_chunk = np.asarray(board_frames[:, list(eeg_channel_indices)], dtype=np.float32)
        projected = np.empty((eeg_chunk.shape[0], len(self.physical_positions)), dtype=np.float32)
        for pos, reve_idx in enumerate(self.physical_positions):
            projected[:, pos] = eeg_chunk[:, self.project_specs[reve_idx][0]]
        return projected

    def _synthesize_full(self, cleaned_physical):
        """Rebuilds the full REVE layout: physical channels in place, midline
        synthesized from the CLEANED lateral pairs (midline carries no independent
        signal, so it must never pass through ICA as its own channel)."""
        n = cleaned_physical.shape[0]
        full = np.empty((n, len(REVE_CHANNEL_NAMES)), dtype=np.float32)
        for pos, reve_idx in enumerate(self.physical_positions):
            full[:, reve_idx] = cleaned_physical[:, pos]
        for reve_idx, source_positions in self.synthesized_outputs:
            full[:, reve_idx] = np.mean(cleaned_physical[:, list(source_positions)], axis=1)
        return full

    def _initialize_filter_states(self, channel_first):
        if self.bandpass_sos is not None and self.bandpass_states is None:
            template = sosfilt_zi(self.bandpass_sos)
            self.bandpass_states = [
                template * float(channel_first[idx, 0])
                for idx in range(channel_first.shape[0])
            ]

        if self.notch_coeffs is not None and self.notch_states is None:
            b_coeffs, a_coeffs = self.notch_coeffs
            template = lfilter_zi(b_coeffs, a_coeffs)
            self.notch_states = [
                template * float(channel_first[idx, 0])
                for idx in range(channel_first.shape[0])
            ]

    def _apply_causal_filters(self, samples):
        channel_first = np.asarray(samples, dtype=np.float32).T
        if channel_first.ndim != 2 or channel_first.shape[1] == 0:
            return np.asarray(samples, dtype=np.float32)

        self._initialize_filter_states(channel_first)
        filtered = np.empty_like(channel_first)
        for idx in range(channel_first.shape[0]):
            signal = channel_first[idx].astype(np.float32, copy=True)

            if self.notch_coeffs is not None:
                b_coeffs, a_coeffs = self.notch_coeffs
                signal, self.notch_states[idx] = lfilter(
                    b_coeffs,
                    a_coeffs,
                    signal,
                    zi=self.notch_states[idx],
                )

            if self.bandpass_sos is not None:
                signal, self.bandpass_states[idx] = sosfilt(
                    self.bandpass_sos,
                    signal,
                    zi=self.bandpass_states[idx],
                )

            filtered[idx] = signal

        # CAR and clipping deliberately do NOT happen here anymore: CAR would make
        # the physical-channel matrix rank-deficient before ICA, and clipping is a
        # per-channel nonlinearity that belongs after reconstruction.
        return filtered.T.astype(np.float32, copy=False)

    def _append_ica_history(self, samples, ocular_reference=None):
        if not self.enable_ica:
            return

        sample_array = np.asarray(samples, dtype=np.float32)
        if sample_array.ndim != 2 or sample_array.shape[0] == 0:
            return

        n = int(sample_array.shape[0])
        self.samples_seen += n

        if ocular_reference is None:
            ocular_values = np.full(n, np.nan, dtype=np.float32)
        else:
            ocular_values = np.asarray(ocular_reference, dtype=np.float32).reshape(-1)
            if ocular_values.shape[0] != n:
                ocular_values = np.full(n, np.nan, dtype=np.float32)

        for row_idx in range(n):
            self.ica_history.append(np.array(sample_array[row_idx], dtype=np.float32, copy=True))
            self.ocular_history.append(float(ocular_values[row_idx]))

    def _fit_ica_if_needed(self):
        if not self.enable_ica:
            return
        if len(self.ica_history) < self.fit_ica_sample_count:
            return
        if self.ica_model is not None and self.ica_refit_disabled:
            return  # frozen: the cleaning function must be identical across load blocks
        if (
            self.ica_model is not None
            and self.last_ica_fit_sample is not None
            and self.refit_ica_sample_count is not None
            and (self.samples_seen - self.last_ica_fit_sample) < self.refit_ica_sample_count
        ):
            return

        fit_history = np.asarray(self.ica_history, dtype=np.float32)[-self.fit_ica_sample_count:]
        n_channels = fit_history.shape[1]  # physical channels only — full rank
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                ica_model = FastICA(
                    n_components=n_channels,
                    random_state=self.ica_random_state,
                    max_iter=self.ica_max_iter,
                    tol=5e-3,
                    whiten="unit-variance",
                )
                sources = ica_model.fit_transform(fit_history)
        except Exception as error:
            print(f"OpenBCIInterface: ICA fit skipped because fitting failed: {error}")
            return

        source_scales = np.std(sources, axis=0) + 1e-6
        source_kurtosis = kurtosis(sources, axis=0, fisher=False, bias=False, nan_policy="omit")
        source_kurtosis = np.nan_to_num(source_kurtosis, nan=0.0, posinf=0.0, neginf=0.0)
        kurtosis_z = (source_kurtosis - np.mean(source_kurtosis)) / (np.std(source_kurtosis) + 1e-6)
        kurtosis_z = np.nan_to_num(kurtosis_z, nan=0.0, posinf=0.0, neginf=0.0)

        mixing = np.abs(np.asarray(ica_model.mixing_, dtype=float))
        mixing = np.nan_to_num(mixing, nan=0.0, posinf=0.0, neginf=0.0)
        frontal_rows = [idx for idx in self.physical_frontal_positions if idx < mixing.shape[0]]
        other_rows = [idx for idx in range(mixing.shape[0]) if idx not in frontal_rows]
        frontal_strength = np.mean(mixing[frontal_rows, :], axis=0) if frontal_rows else np.ones(n_channels)
        other_strength = np.mean(mixing[other_rows, :], axis=0) + 1e-6 if other_rows else 1.0
        frontal_ratio = frontal_strength / other_strength
        frontal_ratio = np.nan_to_num(frontal_ratio, nan=0.0, posinf=0.0, neginf=0.0)
        frontal_gate = frontal_ratio >= self.ica_frontal_ratio_threshold

        # Ocular evidence: correlate each component's time course with the
        # concurrent ocular activity trace (gaze velocity / validity dropouts).
        # Rejecting on spectral/spatial heuristics alone would delete frontal
        # midline theta — the very signal the WM-load classifier reads.
        ocular = np.asarray(self.ocular_history, dtype=np.float32)[-self.fit_ica_sample_count:]
        finite = np.isfinite(ocular)
        ocular_usable = (
            ocular.shape[0] == sources.shape[0]
            and float(finite.mean()) > 0.5
            and float(np.nanstd(ocular)) > 1e-6
        )

        if ocular_usable:
            reference = np.where(finite, ocular, 0.0).astype(float)
            reference = reference - np.mean(reference)
            reference_norm = float(np.linalg.norm(reference)) + 1e-12
            centered_sources = sources - np.mean(sources, axis=0, keepdims=True)
            source_norms = np.linalg.norm(centered_sources, axis=0) + 1e-12
            ocular_corr = np.abs(centered_sources.T @ reference) / (source_norms * reference_norm)
            component_mask = frontal_gate & (ocular_corr >= self.ocular_corr_threshold)
        else:
            if not self.ocular_fallback_warned:
                print(
                    "OpenBCIInterface: no ocular reference available — falling back to a "
                    "conservative rejection rule (frontal + extreme kurtosis only). "
                    "Provide gaze data for validated ocular component removal."
                )
                self.ocular_fallback_warned = True
            component_mask = frontal_gate & (np.abs(kurtosis_z) >= self.ica_kurtosis_z_threshold)

        self.ica_model = ica_model
        self.ica_component_mask = component_mask
        self.ica_source_limits = self.ica_source_z_threshold * source_scales
        self.last_ica_fit_sample = self.samples_seen
        if not self.ica_ready_logged:
            print(
                "OpenBCIInterface: ICA is ready and FROZEN for the session "
                f"({n_channels} physical-channel components, suppressing {int(component_mask.sum())} "
                f"ocular components, ocular evidence {'used' if ocular_usable else 'unavailable'})."
            )
            self.ica_ready_logged = True
        else:
            print(
                "OpenBCIInterface: ICA was refreshed "
                f"(suppressing {int(component_mask.sum())} artifact components)."
            )

    def _apply_ica(self, samples):
        if not self.enable_ica or self.ica_model is None:
            return np.asarray(samples, dtype=np.float32)

        dsp_samples = np.asarray(samples, dtype=np.float32)
        try:
            sources = self.ica_model.transform(dsp_samples)
        except Exception:
            return dsp_samples

        if self.ica_component_mask is not None and self.ica_component_mask.any():
            sources[:, self.ica_component_mask] = 0.0

        if self.ica_source_limits is not None:
            sources = np.clip(sources, -self.ica_source_limits, self.ica_source_limits)

        try:
            reconstructed = self.ica_model.inverse_transform(sources)
        except Exception:
            return dsp_samples
        return np.array(reconstructed, dtype=np.float32, copy=False)

    def _smooth_artifact_weight(self, reference_z):
        over_threshold = np.abs(reference_z) - self.frontal_artifact_z_threshold
        if np.max(over_threshold) <= 0:
            return np.zeros_like(reference_z, dtype=np.float32)

        weight = np.clip(over_threshold / max(self.frontal_artifact_z_threshold, 1e-6), 0.0, 1.0)
        kernel_len = self.artifact_kernel_len
        if kernel_len > 1 and weight.size > 1:
            kernel = np.hanning(kernel_len)
            if np.allclose(kernel.sum(), 0.0):
                kernel = np.ones(kernel_len, dtype=float)
            kernel = kernel / np.sum(kernel)
            pad = kernel_len // 2
            padded = np.pad(weight, (pad, pad), mode="edge")
            weight = np.convolve(padded, kernel, mode="same")[pad:-pad]
        return np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)

    def _attenuate_frontal_artifacts(self, samples):
        cleaned = np.asarray(samples, dtype=np.float32).copy()
        # Off by default: this stage triggers on frontal excursions, which grow
        # with WM load (more eye movement) — enabling it makes cleaning intensity
        # a load-correlated confound. Opt in only for non-calibration use.
        if self.frontal_regression_strength <= 0.0:
            return cleaned
        if cleaned.ndim != 2 or cleaned.shape[0] < 4 or not self.frontal_indices:
            return cleaned

        frontal_signal = np.mean(cleaned[:, self.frontal_indices], axis=1)
        if self.posterior_indices:
            posterior_signal = np.mean(cleaned[:, self.posterior_indices], axis=1)
            artifact_reference = frontal_signal - posterior_signal
        else:
            artifact_reference = frontal_signal

        median = float(np.median(artifact_reference))
        mad = float(np.median(np.abs(artifact_reference - median)))
        sigma = 1.4826 * mad
        if sigma <= 1e-6:
            sigma = float(np.std(artifact_reference)) + 1e-6

        reference_centered = artifact_reference - median
        reference_z = reference_centered / sigma
        artifact_weight = self._smooth_artifact_weight(reference_z)
        if np.max(artifact_weight) <= 1e-3:
            return cleaned

        weighted_reference = artifact_weight * reference_centered
        reference_variance = float(np.dot(weighted_reference, weighted_reference)) + 1e-6
        for channel_idx in range(cleaned.shape[1]):
            channel = cleaned[:, channel_idx]
            slope = float(np.dot(channel, weighted_reference) / reference_variance)
            if channel_idx in self.frontal_indices:
                spatial_scale = 0.70
            elif channel_idx in self.posterior_indices:
                spatial_scale = 0.12
            else:
                spatial_scale = 0.30
            correction_gain = np.clip(
                self.frontal_regression_strength * spatial_scale * artifact_weight,
                0.0,
                self.max_artifact_attenuation,
            )
            cleaned[:, channel_idx] = channel - (correction_gain * slope * reference_centered)

        return cleaned

    def process_chunk(self, board_frames, eeg_channel_indices, ocular_reference=None):
        """Cleans one chunk. `ocular_reference` is an optional per-sample ocular
        activity trace (same length as the chunk; e.g. gaze velocity + validity
        dropouts from the Varjo stream) used as ground truth for ICA rejection."""
        if board_frames.size == 0:
            return np.empty((0, len(REVE_CHANNEL_NAMES)), dtype=np.float32)

        physical = self._project_physical(board_frames, eeg_channel_indices)
        filtered = self._apply_causal_filters(physical)
        self._append_ica_history(filtered, ocular_reference)
        self._fit_ica_if_needed()
        cleaned_physical = self._apply_ica(filtered)

        full = self._synthesize_full(cleaned_physical)
        full = common_average_reference(full.T).T
        full = robust_clip_channels(full.T, n_sigmas=self.robust_clip_sigma).T
        cleaned = self._attenuate_frontal_artifacts(full)
        return cleaned.astype(np.float32, copy=False)


class OpenBCIToLSLInterface:
    _BOARD_ALIASES = {
        "cyton": BoardIds.CYTON_BOARD.value,
        "cyton8": BoardIds.CYTON_BOARD.value,
        "cytondaisy": BoardIds.CYTON_DAISY_BOARD.value,
        "daisy": BoardIds.CYTON_DAISY_BOARD.value,
    }
    _PORT_HINTS = ("openbci", "usb serial", "ftdi", "cp210", "ch340")

    def __init__(
        self,
        stream_name,
        stream_type="EEG",
        preprocessed_stream_name=None,
        serial_port="COM5",
        board_id=BoardIds.CYTON_BOARD.value,
        log=False,
        streamer_params="",
        ring_buffer_size=45000,
        eeg_labels_override=None,
    ):
        """`eeg_labels_override`: the cap's ACTUAL electrode montage (one 10-20
        label per EEG channel, board order). BrainFlow's board descriptor only
        provides the DEFAULT assumed montage — if the physical cap places
        electrodes elsewhere (e.g. real Fz/Cz/Pz on the midline), declare it here
        so the preprocessing and the REVE projection use the true positions
        instead of synthesizing them."""
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.params.ip_port = 0
        self.params.mac_address = ""
        self.params.other_info = ""
        self.params.serial_number = ""
        self.params.ip_address = ""
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = ""

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.preprocessed_stream_name = preprocessed_stream_name or f"{stream_name}_Preprocessed"
        self.board_id = self._resolve_board_id(board_id)
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size
        self.board_descr = BoardShim.get_board_descr(self.board_id)
        self.eeg_labels_override = list(eeg_labels_override) if eeg_labels_override else None
        if self.eeg_labels_override is not None:
            board_eeg_count = len(self.board_descr.get("eeg_channels", []))
            if len(self.eeg_labels_override) != board_eeg_count:
                raise ValueError(
                    f"eeg_labels_override has {len(self.eeg_labels_override)} labels but the "
                    f"{self.board_descr['name']} board records {board_eeg_count} EEG channels."
                )
        self.raw_channel_descriptors = self._build_channel_descriptors()
        self.raw_eeg_labels = [label for label, _, channel_type in self.raw_channel_descriptors if channel_type == "EEG"]
        self.raw_eeg_indices = tuple(
            idx for idx, (_, _, channel_type) in enumerate(self.raw_channel_descriptors) if channel_type == "EEG"
        )
        self.channel_count = 0
        self.info_eeg = None
        self.outlet_eeg = None
        self.info_preprocessed_eeg = None
        self.outlet_preprocessed_eeg = None
        self.preprocessed_channel_count = len(REVE_CHANNEL_NAMES)
        self.eeg_preprocessor = RealTimeEegPreprocessor(
            sampling_rate=self.board_descr["sampling_rate"],
            raw_eeg_labels=self.raw_eeg_labels,
        )
        self.gaze_reference = GazeOcularReferenceInlet()

        if log:
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        try:
            self.board = BoardShim(self.board_id, self.params)
        except brainflow.board_shim.BrainFlowError as error:
            raise RuntimeError(
                f"Cannot create a BrainFlow board for board_id={self.board_id} on {serial_port}."
            ) from error

    @classmethod
    def _resolve_board_id(cls, board_id):
        if isinstance(board_id, int):
            return board_id

        if isinstance(board_id, str):
            stripped = board_id.strip()
            if stripped.lstrip("-").isdigit():
                return int(stripped)

            normalized = "".join(character for character in stripped.lower() if character.isalnum())
            if normalized in cls._BOARD_ALIASES:
                return cls._BOARD_ALIASES[normalized]

        raise ValueError(
            f"Unsupported board_id {board_id!r}. Use 0/'cyton' or 2/'cyton+daisy'."
        )

    @staticmethod
    def _get_serial_ports():
        return list(list_ports.comports())

    @classmethod
    def _format_serial_ports(cls):
        ports = cls._get_serial_ports()
        if not ports:
            return "No serial ports detected."

        lines = []
        for port in ports:
            parts = [port.device, port.description]
            if port.manufacturer:
                parts.append(f"manufacturer={port.manufacturer}")
            if port.serial_number:
                parts.append(f"serial={port.serial_number}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    @classmethod
    def _find_openbci_port(cls):
        ports = cls._get_serial_ports()
        for port in ports:
            searchable = " ".join(
                value for value in (port.device, port.description, port.manufacturer or "") if value
            ).lower()
            if any(hint in searchable for hint in cls._PORT_HINTS):
                return port.device
        return ports[0].device if ports else None

    def _resolve_serial_port(self):
        port = (self.params.serial_port or "").strip()
        if port and port.upper() != "AUTO":
            return port

        detected_port = self._find_openbci_port()
        if detected_port is None:
            raise RuntimeError(
                "Unable to detect an OpenBCI serial port.\n"
                f"Detected ports:\n{self._format_serial_ports()}"
            )
        self.params.serial_port = detected_port
        return detected_port

    def _assert_serial_port_accessible(self):
        port = self._resolve_serial_port()
        try:
            probe = serial.Serial(port=port, timeout=1)
            probe.close()
        except serial.SerialException as error:
            error_message = str(error)
            if "Access is denied" in error_message:
                raise RuntimeError(
                    f"Serial port {port} exists but is already in use by another application.\n"
                    "Close any program that may still own the dongle, such as another Python process, "
                    "OpenBCI GUI, Arduino Serial Monitor, PuTTY, or a previous PhysioLabXR run.\n"
                    f"Detected ports:\n{self._format_serial_ports()}"
                ) from error
            raise RuntimeError(
                f"Unable to open serial port {port}: {error_message}\n"
                f"Detected ports:\n{self._format_serial_ports()}"
            ) from error

    def _build_channel_descriptors(self):
        eeg_labels = self.eeg_labels_override or [
            label.strip()
            for label in self.board_descr.get("eeg_names", "").split(",")
            if label.strip()
        ]
        accel_count = len(self.board_descr.get("accel_channels", []))
        other_count = len(self.board_descr.get("other_channels", []))
        analog_count = len(self.board_descr.get("analog_channels", []))

        descriptors = [("PackageNum", "counter", "COUNTER")]
        descriptors.extend((label, "microvolts", "EEG") for label in eeg_labels)

        if accel_count == 3:
            descriptors.extend(
                (label, "g", "ACC") for label in ("X", "Y", "Z")
            )
        else:
            descriptors.extend(
                (f"Accel{index + 1}", "g", "ACC") for index in range(accel_count)
            )

        descriptors.extend(
            (f"Other{index + 1}", "aux", "AUX") for index in range(other_count)
        )
        descriptors.extend(
            (f"Analog{index + 1}", "aux", "AUX") for index in range(analog_count)
        )
        descriptors.append(("TimeStamp", "seconds", "TIME"))
        descriptors.append(("Marker", "marker", "MARKER"))
        return descriptors

    @staticmethod
    def _build_preprocessed_channel_descriptors():
        return [(label, "microvolts", "EEG") for label in REVE_CHANNEL_NAMES]

    def start_sensor(self):
        self._assert_serial_port_accessible()
        try:
            self.board.prepare_session()
        except brainflow.board_shim.BrainFlowError as error:
            raise AssertionError(
                "Unable to connect to device: please check the sensor connection, the COM port, and the board_id."
            ) from error
        print(
            f"OpenBCIInterface: connected to sensor on {self.params.serial_port} with board id {self.board_id}."
        )

        try:
            self.board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError as error:
            raise AssertionError(
                "Unable to start streaming: please verify the OpenBCI board is powered on and the board_id matches the hardware."
            ) from error
        print("OpenBCIInterface: started streaming.")

        self.create_lsl()

    def process_frames(self):
        frames = self.board.get_board_data()
        if frames.size == 0:
            return frames

        raw_chunk = np.asarray(frames.T, dtype=np.float32)
        self.push_chunk(raw_chunk)

        ocular_reference = self.gaze_reference.reference_for(
            raw_chunk.shape[0], self.board_descr["sampling_rate"])
        preprocessed_chunk = self.eeg_preprocessor.process_chunk(
            raw_chunk, self.raw_eeg_indices, ocular_reference=ocular_reference)
        if preprocessed_chunk.size:
            self.push_preprocessed_chunk(preprocessed_chunk)

        return frames

    def stop_sensor(self):
        try:
            if self.board.is_prepared():
                try:
                    self.board.stop_stream()
                    print("OpenBCIInterface: stopped streaming.")
                except brainflow.board_shim.BrainFlowError:
                    pass

                self.board.release_session()
                print("OpenBCIInterface: released session.")
        except brainflow.board_shim.BrainFlowError as error:
            print(error)

    def create_lsl(self, name=None, type=None, nominal_srate=None, channel_format="float32", source_id=None):
        name = name or self.stream_name
        stream_type = type or self.stream_type
        nominal_srate = nominal_srate or self.board_descr["sampling_rate"]
        self.channel_count = self.board_descr["num_rows"]
        source_id = source_id or f"OpenBCI_{self.board_descr['name']}_{self.board_id}"

        self.info_eeg = StreamInfo(
            name=name,
            type=stream_type,
            channel_count=self.channel_count,
            nominal_srate=nominal_srate,
            channel_format=channel_format,
            source_id=source_id,
        )

        chns = self.info_eeg.desc().append_child("channels")
        channel_descriptors = self.raw_channel_descriptors
        if len(channel_descriptors) != self.channel_count:
            raise RuntimeError(
                f"Board description generated {len(channel_descriptors)} labels for a {self.channel_count}-channel stream."
            )

        for label, unit, channel_type in channel_descriptors:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", unit)
            ch.append_child_value("type", channel_type)

        self.info_eeg.desc().append_child_value("manufacturer", "OpenBCI Inc.")
        self.outlet_eeg = StreamOutlet(self.info_eeg)

        preprocessed_source_id = f"{source_id}_preprocessed"
        self.info_preprocessed_eeg = StreamInfo(
            name=self.preprocessed_stream_name,
            type=stream_type,
            channel_count=self.preprocessed_channel_count,
            nominal_srate=nominal_srate,
            channel_format=channel_format,
            source_id=preprocessed_source_id,
        )

        preprocessed_channels = self.info_preprocessed_eeg.desc().append_child("channels")
        for label, unit, channel_type in self._build_preprocessed_channel_descriptors():
            ch = preprocessed_channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", unit)
            ch.append_child_value("type", channel_type)

        self.info_preprocessed_eeg.desc().append_child_value("manufacturer", "OpenBCI Inc.")
        preprocessing_tag = (
            "realtime_bandpass_car_fastica_soft_ocular_suppression"
            if self.eeg_preprocessor.enable_ica
            else "realtime_bandpass_car_soft_ocular_suppression"
        )
        self.info_preprocessed_eeg.desc().append_child_value("preprocessing", preprocessing_tag)
        self.outlet_preprocessed_eeg = StreamOutlet(self.info_preprocessed_eeg)

        print(
            "--------------------------------------\n"
            "LSL Configuration:\n"
            "  Stream 1:\n"
            f"      Name: {name}\n"
            f"      Type: {stream_type}\n"
            f"      Board Profile: {self.board_descr['name']} ({self.board_id})\n"
            f"      Channel Count: {self.channel_count}\n"
            f"      Sampling Rate: {nominal_srate}\n"
            f"      Channel Format: {channel_format}\n"
            f"      Source Id: {source_id}\n"
            "  Stream 2:\n"
            f"      Name: {self.preprocessed_stream_name}\n"
            f"      Type: {stream_type}\n"
            f"      Channel Count: {self.preprocessed_channel_count}\n"
            f"      Sampling Rate: {nominal_srate}\n"
            f"      Channel Format: {channel_format}\n"
            f"      Source Id: {preprocessed_source_id}\n"
        )

    def push_chunk(self, samples):
        if self.outlet_eeg is None:
            raise RuntimeError("LSL outlet is not ready. Call start_sensor() or create_lsl() first.")

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.channel_count:
            raise ValueError(
                f"Raw LSL chunk width mismatch: got {array.shape[1]} values, "
                f"but the outlet expects {self.channel_count}."
            )
        self.outlet_eeg.push_chunk(array.tolist())

    def push_preprocessed_chunk(self, samples):
        if self.outlet_preprocessed_eeg is None:
            raise RuntimeError("Preprocessed LSL outlet is not ready. Call start_sensor() or create_lsl() first.")

        array = np.asarray(samples, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape[1] != self.preprocessed_channel_count:
            raise ValueError(
                f"Preprocessed LSL chunk width mismatch: got {array.shape[1]} values, "
                f"but the outlet expects {self.preprocessed_channel_count}."
            )
        self.outlet_preprocessed_eeg.push_chunk(array.tolist())

    def info_print(self):
        print("Board Information:")
        print("Sampling Rate:", self.board_descr["sampling_rate"])
        print("Board Id:", self.board_id)
        print("EEG names:", self.board_descr["eeg_names"])
        print("Package Num Channel: ", self.board_descr["package_num_channel"])
        print("EEG Channels:", self.board_descr["eeg_channels"])
        print("Accel Channels: ", self.board_descr["accel_channels"])
        print("Other Channels:", self.board_descr["other_channels"])
        print("Analog Channels: ", self.board_descr["analog_channels"])
        print("TimeStamp: ", self.board_descr["timestamp_channel"])
        print("Marker Channel: ", self.board_descr["marker_channel"])


def run_test_lsl(openbci_interface):
    while True:
        try:
            openbci_interface.process_frames()
        except KeyboardInterrupt:
            print("Stopped streaming.")
            break
    return True


def main(stream_name, stream_type, preprocessed_stream_name, serial_port, board_id, log,
         ring_buffer_size, eeg_labels_override=None):
    openbci_interface = OpenBCIToLSLInterface(
        stream_name=stream_name,
        stream_type=stream_type,
        preprocessed_stream_name=preprocessed_stream_name,
        serial_port=serial_port,
        board_id=board_id,
        log=log,
        ring_buffer_size=ring_buffer_size,
        eeg_labels_override=eeg_labels_override,
    )

    try:
        openbci_interface.start_sensor()
        run_test_lsl(openbci_interface)
    finally:
        openbci_interface.stop_sensor()


if __name__ == "__main__":
    main(
        stream_name="OpenBCI_Cyton_Daisy_15_Channels",
        stream_type="EEG",
        preprocessed_stream_name="OpenBCI_Cyton_Daisy_15_Channels_Preprocessed",
        serial_port="COM7",
        board_id="cyton+daisy",
        log=False,
        ring_buffer_size=45000,
        eeg_labels_override=list(CAP_MONTAGE),
    )
