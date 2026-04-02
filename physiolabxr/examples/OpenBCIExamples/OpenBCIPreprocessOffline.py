import time
from collections import deque

import numpy as np
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
from scipy.signal import butter, detrend, filtfilt, iirnotch, sosfiltfilt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA


REVE_CHANNEL_NAMES = (
    "Fp1", "Fp2", "F7", "F3", "Fz",
    "F4", "F8", "C3", "Cz", "C4",
    "P3", "Pz", "P4", "O1", "O2",
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


def bandpass(samples, fs_hz, low_hz, high_hz):
    if fs_hz <= 0 or samples.shape[-1] < 16:
        return samples

    nyquist = fs_hz / 2.0
    high = min(high_hz, nyquist * 0.95)
    low = max(low_hz, 0.01)
    if low >= high:
        return samples

    sos = butter(4, [low / nyquist, high / nyquist], btype="band", output="sos")
    try:
        return sosfiltfilt(sos, samples, axis=-1)
    except ValueError:
        return samples


def notch_filter(samples, fs_hz, notch_hz, quality=30.0):
    if fs_hz <= 0 or notch_hz <= 0 or samples.shape[-1] < 16:
        return samples

    nyquist = fs_hz / 2.0
    if notch_hz >= nyquist * 0.95:
        return samples

    try:
        b_coeffs, a_coeffs = iirnotch(notch_hz, quality, fs=fs_hz)
        return filtfilt(b_coeffs, a_coeffs, samples, axis=-1)
    except ValueError:
        return samples


def common_average_reference(samples):
    if samples.ndim != 2 or samples.shape[0] < 2:
        return samples
    return samples - np.mean(samples, axis=0, keepdims=True)


def robust_clip_channels(samples, n_sigmas=5.0):
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


def extract_channel_descriptors(stream_info):
    descriptors = []
    channels_node = stream_info.desc().child("channels")
    if channels_node.empty():
        return descriptors

    channel_node = channels_node.child("channel")
    while not channel_node.empty():
        descriptors.append(
            (
                channel_node.child_value("label") or "",
                channel_node.child_value("unit") or "",
                channel_node.child_value("type") or "",
            )
        )
        channel_node = channel_node.next_sibling("channel")
    return descriptors


class OfflineEegPreprocessor:
    def __init__(
        self,
        *,
        sampling_rate,
        raw_eeg_labels,
        low_hz=1.0,
        high_hz=40.0,
        line_noise_hz=60.0,
        history_sec=60.0,
        fit_ica_sec=30.0,
        refit_ica_sec=20.0,
        ica_max_iter=1000,
        ica_random_state=42,
        robust_clip_sigma=5.0,
        ica_source_z_threshold=2.75,
        ica_kurtosis_z_threshold=1.75,
        ica_frontal_ratio_threshold=1.10,
        ica_low_freq_ratio_threshold=0.50,
        frontal_artifact_z_threshold=2.60,
        frontal_regression_strength=0.45,
        artifact_interp_half_width_sec=0.16,
        max_artifact_attenuation=0.55,
    ):
        self.sampling_rate = float(sampling_rate)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.line_noise_hz = float(line_noise_hz)
        self.fit_ica_sec = float(fit_ica_sec)
        self.refit_ica_sec = float(refit_ica_sec)
        self.ica_max_iter = int(ica_max_iter)
        self.ica_random_state = int(ica_random_state)
        self.robust_clip_sigma = float(robust_clip_sigma)
        self.ica_source_z_threshold = float(ica_source_z_threshold)
        self.ica_kurtosis_z_threshold = float(ica_kurtosis_z_threshold)
        self.ica_frontal_ratio_threshold = float(ica_frontal_ratio_threshold)
        self.ica_low_freq_ratio_threshold = float(ica_low_freq_ratio_threshold)
        self.frontal_artifact_z_threshold = float(frontal_artifact_z_threshold)
        self.frontal_regression_strength = float(frontal_regression_strength)
        self.max_artifact_attenuation = float(max_artifact_attenuation)
        self.artifact_interp_half_width = max(
            int(round(float(artifact_interp_half_width_sec) * self.sampling_rate)),
            1,
        )

        max_history = max(int(round(history_sec * self.sampling_rate)), 1)
        self.history = deque(maxlen=max_history)
        self.fit_sample_count = max(int(round(self.fit_ica_sec * self.sampling_rate)), 256)
        self.refit_sample_count = max(int(round(self.refit_ica_sec * self.sampling_rate)), 1)
        self.project_specs = self._build_project_specs(raw_eeg_labels)
        self.frontal_indices = tuple(
            idx for idx, label in enumerate(REVE_CHANNEL_NAMES) if label in FRONTAL_CHANNELS
        )
        self.posterior_indices = tuple(
            idx for idx, label in enumerate(REVE_CHANNEL_NAMES) if label in POSTERIOR_CHANNELS
        )

        self.ica_model = None
        self.ica_component_mask = None
        self.ica_source_limits = None
        self.ica_ready_logged = False
        self.samples_seen = 0
        self.last_ica_fit_sample = None

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
                raise RuntimeError(f"Cannot map required REVE channel `{label}` from raw labels.")
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

    def _project_eeg_sample(self, eeg_values):
        projected = []
        for spec in self.project_specs:
            if len(spec) == 1:
                projected.append(float(eeg_values[spec[0]]))
            else:
                projected.append(float(np.mean([eeg_values[index] for index in spec])))
        return np.asarray(projected, dtype=np.float32)

    def _apply_dsp(self, samples):
        channel_first = np.asarray(samples, dtype=np.float32).T
        try:
            channel_first = detrend(channel_first, axis=-1, type="linear").astype(np.float32, copy=False)
        except ValueError:
            pass

        channel_first = notch_filter(channel_first, self.sampling_rate, self.line_noise_hz)
        channel_first = bandpass(channel_first, self.sampling_rate, self.low_hz, self.high_hz)
        channel_first = common_average_reference(channel_first)
        channel_first = robust_clip_channels(channel_first, n_sigmas=self.robust_clip_sigma)
        return channel_first.T.astype(np.float32, copy=False)

    def _fit_ica(self, dsp_history):
        if dsp_history.shape[0] < self.fit_sample_count:
            return

        if (
            self.ica_model is not None
            and self.last_ica_fit_sample is not None
            and (self.samples_seen - self.last_ica_fit_sample) < self.refit_sample_count
        ):
            return

        n_channels = dsp_history.shape[1]
        fit_history = dsp_history[-self.fit_sample_count:]
        try:
            ica_model = FastICA(
                n_components=n_channels,
                random_state=self.ica_random_state,
                max_iter=self.ica_max_iter,
                tol=5e-3,
                whiten="unit-variance",
            )
            sources = ica_model.fit_transform(fit_history)
        except Exception as error:
            print(f"OfflinePreprocess: ICA fit skipped because fitting failed: {error}")
            return

        source_scales = np.std(sources, axis=0) + 1e-6
        source_kurtosis = kurtosis(sources, axis=0, fisher=False, bias=False, nan_policy="omit")
        kurtosis_z = (source_kurtosis - np.mean(source_kurtosis)) / (np.std(source_kurtosis) + 1e-6)
        spectra = np.abs(np.fft.rfft(sources, axis=0)) ** 2
        freqs = np.fft.rfftfreq(sources.shape[0], d=1.0 / self.sampling_rate)
        low_band_mask = (freqs >= 0.5) & (freqs <= 4.0)
        workload_band_mask = (freqs >= 0.5) & (freqs <= min(self.high_hz, 25.0))
        if low_band_mask.any() and workload_band_mask.any():
            low_band_power = np.mean(spectra[low_band_mask, :], axis=0)
            workload_band_power = np.mean(spectra[workload_band_mask, :], axis=0) + 1e-6
            low_freq_ratio = low_band_power / workload_band_power
        else:
            low_freq_ratio = np.zeros(n_channels, dtype=float)

        frontal_indices = [idx for idx in self.frontal_indices if idx < ica_model.mixing_.shape[0]]
        other_indices = [idx for idx in range(ica_model.mixing_.shape[0]) if idx not in frontal_indices]
        mixing = np.abs(np.asarray(ica_model.mixing_, dtype=float))
        frontal_strength = np.mean(mixing[frontal_indices, :], axis=0) if frontal_indices else np.ones(n_channels)
        other_strength = np.mean(mixing[other_indices, :], axis=0) + 1e-6 if other_indices else 1.0
        frontal_ratio = frontal_strength / other_strength

        component_mask = (
            (
                (np.abs(kurtosis_z) >= self.ica_kurtosis_z_threshold)
                | (low_freq_ratio >= self.ica_low_freq_ratio_threshold)
            )
            & (frontal_ratio >= self.ica_frontal_ratio_threshold)
        )

        self.ica_model = ica_model
        self.ica_component_mask = component_mask
        self.ica_source_limits = self.ica_source_z_threshold * source_scales
        self.last_ica_fit_sample = self.samples_seen
        if not self.ica_ready_logged:
            print(
                "OfflinePreprocess: ICA is ready "
                f"({n_channels} components, suppressing {int(component_mask.sum())} artifact components)."
            )
            self.ica_ready_logged = True
        else:
            print(
                "OfflinePreprocess: ICA was refreshed "
                f"(suppressing {int(component_mask.sum())} artifact components)."
            )

    def _apply_ica(self, dsp_samples):
        if self.ica_model is None:
            return dsp_samples

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
        return np.asarray(reconstructed, dtype=np.float32)

    def _smooth_artifact_weight(self, reference_z):
        over_threshold = np.abs(reference_z) - self.frontal_artifact_z_threshold
        if np.max(over_threshold) <= 0:
            return np.zeros_like(reference_z, dtype=np.float32)

        weight = np.clip(over_threshold / max(self.frontal_artifact_z_threshold, 1e-6), 0.0, 1.0)
        window = (2 * self.artifact_interp_half_width) + 1
        if window > 1:
            kernel = np.hanning(window)
            if np.allclose(kernel.sum(), 0.0):
                kernel = np.ones(window, dtype=float)
            kernel = kernel / np.sum(kernel)
            pad = self.artifact_interp_half_width
            padded = np.pad(weight, (pad, pad), mode="edge")
            weight = np.convolve(padded, kernel, mode="same")[pad:-pad]
        return np.clip(weight, 0.0, 1.0).astype(np.float32, copy=False)

    def _suppress_frontal_artifacts(self, samples):
        cleaned = np.asarray(samples, dtype=np.float32).copy()
        if cleaned.ndim != 2 or cleaned.shape[0] < max(int(round(1.0 * self.sampling_rate)), 16):
            return cleaned
        if not self.frontal_indices:
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
            return cleaned

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
                spatial_scale = 1.0
            elif channel_idx in self.posterior_indices:
                spatial_scale = 0.20
            else:
                spatial_scale = 0.45
            correction_gain = np.clip(
                self.frontal_regression_strength * spatial_scale * artifact_weight,
                0.0,
                self.max_artifact_attenuation,
            )
            cleaned[:, channel_idx] = channel - (correction_gain * slope * reference_centered)

        return cleaned

    def process_chunk(self, input_samples, eeg_channel_indices):
        if input_samples.size == 0:
            return np.empty((0, len(REVE_CHANNEL_NAMES)), dtype=np.float32)

        self.samples_seen += int(input_samples.shape[0])
        new_samples = []
        for frame in input_samples:
            eeg_values = frame[list(eeg_channel_indices)]
            projected = self._project_eeg_sample(eeg_values)
            self.history.append(projected)
            new_samples.append(projected)

        if not new_samples or len(self.history) < max(int(round(5.0 * self.sampling_rate)), len(new_samples)):
            return np.empty((0, len(REVE_CHANNEL_NAMES)), dtype=np.float32)

        history_array = np.asarray(self.history, dtype=np.float32)
        dsp_history = self._apply_dsp(history_array)
        self._fit_ica(dsp_history)
        clean_history = self._apply_ica(dsp_history)
        clean_history = self._suppress_frontal_artifacts(clean_history)
        return clean_history[-len(new_samples):].astype(np.float32, copy=False)


class LslOfflinePreprocessorBridge:
    def __init__(
        self,
        *,
        input_stream_name,
        output_stream_name,
        input_stream_type="EEG",
        resolve_timeout_sec=10.0,
        max_chunk_samples=512,
    ):
        self.input_stream_name = input_stream_name
        self.output_stream_name = output_stream_name
        self.input_stream_type = input_stream_type
        self.resolve_timeout_sec = float(resolve_timeout_sec)
        self.max_chunk_samples = int(max_chunk_samples)
        self.inlet = None
        self.outlet = None
        self.preprocessor = None
        self.eeg_channel_indices = None

    def _resolve_input_stream(self):
        streams = resolve_byprop("name", self.input_stream_name, timeout=self.resolve_timeout_sec)
        if not streams:
            raise RuntimeError(
                f"Could not find an LSL stream named `{self.input_stream_name}` within "
                f"{self.resolve_timeout_sec:.1f} seconds."
            )

        if self.input_stream_type:
            for stream in streams:
                if stream.type() == self.input_stream_type:
                    return stream
        return streams[0]

    @staticmethod
    def _extract_eeg_layout(stream_info):
        descriptors = extract_channel_descriptors(stream_info)
        if not descriptors:
            channel_count = stream_info.channel_count()
            labels = [f"Ch{index + 1}" for index in range(channel_count)]
            indices = tuple(range(channel_count))
            return labels, indices

        eeg_indices = [
            index
            for index, (_, _, channel_type) in enumerate(descriptors)
            if channel_type.strip().upper() == "EEG"
        ]
        if not eeg_indices:
            eeg_indices = list(range(len(descriptors)))

        eeg_labels = []
        for index in eeg_indices:
            label, _, _ = descriptors[index]
            eeg_labels.append(label or f"Ch{index + 1}")
        return eeg_labels, tuple(eeg_indices)

    def start(self):
        stream_info = self._resolve_input_stream()
        self.inlet = StreamInlet(
            stream_info,
            max_buflen=360,
            max_chunklen=self.max_chunk_samples,
            recover=True,
        )

        sampling_rate = float(stream_info.nominal_srate())
        if sampling_rate <= 0:
            raise RuntimeError("The input LSL stream must provide a positive nominal sampling rate.")

        eeg_labels, eeg_indices = self._extract_eeg_layout(stream_info)
        self.preprocessor = OfflineEegPreprocessor(
            sampling_rate=sampling_rate,
            raw_eeg_labels=eeg_labels,
        )
        self.eeg_channel_indices = eeg_indices

        output_info = StreamInfo(
            name=self.output_stream_name,
            type=self.input_stream_type,
            channel_count=len(REVE_CHANNEL_NAMES),
            nominal_srate=sampling_rate,
            channel_format="float32",
            source_id=f"{stream_info.source_id()}_offline_preprocessed",
        )
        channels = output_info.desc().append_child("channels")
        for label in REVE_CHANNEL_NAMES:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("unit", "microvolts")
            channel.append_child_value("type", "EEG")
        output_info.desc().append_child_value("manufacturer", "OpenBCI Inc.")
        output_info.desc().append_child_value("preprocessing", "offline_dsp_plus_ica")
        output_info.desc().append_child_value("input_stream_name", self.input_stream_name)
        self.outlet = StreamOutlet(output_info)

        print(
            "OfflinePreprocess: connected to input stream.\n"
            f"  Input Name: {stream_info.name()}\n"
            f"  Input Type: {stream_info.type()}\n"
            f"  Sampling Rate: {sampling_rate}\n"
            f"  Output Name: {self.output_stream_name}\n"
            f"  Output Channels: {len(REVE_CHANNEL_NAMES)}"
        )

    def run(self):
        if self.inlet is None or self.outlet is None or self.preprocessor is None:
            self.start()

        while True:
            samples, timestamps = self.inlet.pull_chunk(timeout=0.5, max_samples=self.max_chunk_samples)
            if not timestamps:
                continue

            chunk = np.asarray(samples, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            cleaned = self.preprocessor.process_chunk(chunk, self.eeg_channel_indices)
            if cleaned.size == 0:
                continue

            if len(timestamps) >= len(cleaned):
                out_timestamps = timestamps[-len(cleaned):]
                self.outlet.push_chunk(cleaned.tolist(), out_timestamps)
            else:
                self.outlet.push_chunk(cleaned.tolist())


def main(
    input_stream_name="OpenBCI_Cyton_Daisy_15_Channels",
    output_stream_name="OpenBCI_Cyton_Daisy_15_Channels_Preprocessed_Offline",
    input_stream_type="EEG",
):
    bridge = LslOfflinePreprocessorBridge(
        input_stream_name=input_stream_name,
        output_stream_name=output_stream_name,
        input_stream_type=input_stream_type,
    )
    bridge.start()
    bridge.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("OfflinePreprocess: stopped.")
        time.sleep(0.1)
