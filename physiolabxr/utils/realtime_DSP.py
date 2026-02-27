from multiprocessing import Pool

import numpy as np
from enum import Enum
import numpy as np
from scipy import sparse
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt
from scipy.sparse.linalg import spsolve
from enum import Enum

try:
    from sklearn.decomposition import FastICA
except Exception:
    FastICA = None

class DataProcessorType(Enum):
    RealtimeNotch = 'RealtimeNotch'
    RealtimeButterBandpass = 'RealtimeButterBandpass'
    RealtimeVrms = 'RealtimeVrms'

class DataProcessor:
    def __init__(self, data_processor_type: DataProcessorType):
        self.data_processor_type = data_processor_type
        self.data_processor_activated = False

    def process_sample(self, data):
        return data

    def process_buffer(self, data):
        output_buffer = np.empty(shape=data.shape)
        for index in range(0, data.shape[1]):
            output_buffer[:, index] = self.process_sample(data[:, index])
        return output_buffer

    def reset_tap(self):
        pass

    def activate_data_processor(self):
        pass

# class IIRFilter(DataProcessor):
#
#     def __init__(self):
#         super().__init__()
#         self.a = None
#         self.b = None
#         self.x_tap = None
#         self.y_tap = None
#
#     def process_sample(self, data):
#         # perform realtime filter with tap
#
#         # push x
#         self.x_tap[:, 1:] = self.x_tap[:, : -1]
#         self.x_tap[:, 0] = data
#         # push y
#         self.y_tap[:, 1:] = self.y_tap[:, : -1]
#         # calculate new y
#         self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
#                            np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)
#
#         data = self.y_tap[:, 0]
#         return data
#
#     def reset_tap(self):
#         self.x_tap.fill(0)
#         self.y_tap.fill(0)

class RealtimeNotch(DataProcessor):
    def __init__(self, w0=60, Q=20, fs=250, channel_num=8):
        super().__init__(DataProcessorType.RealtimeNotch)
        self.w0 = w0
        self.Q = Q
        self.fs = fs
        self.channel_num = channel_num
        self.b, self.a = iirnotch(w0=w0, Q=self.Q, fs=self.fs)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = data
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y
        self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                           np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)

        data = self.y_tap[:, 0]
        return data

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)


class RealtimeButterBandpass(DataProcessor):
    def __init__(self, lowcut=5, highcut=50, fs=250, order=5, channel_num=8):
        super().__init__(DataProcessorType.RealtimeButterBandpass)
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.channel_num = channel_num
        self.b, self.a = self.butter_bandpass(lowcut=self.lowcut, highcut=self.highcut, fs=self.fs, order=self.order)
        self.x_tap = np.zeros((self.channel_num, len(self.b)))
        self.y_tap = np.zeros((self.channel_num, len(self.a)))

    def process_sample(self, data):
        # perform realtime filter with tap

        # push x
        self.x_tap[:, 1:] = self.x_tap[:, : -1]
        self.x_tap[:, 0] = data
        # push y
        self.y_tap[:, 1:] = self.y_tap[:, : -1]
        # calculate new y
        self.y_tap[:, 0] = np.sum(np.multiply(self.x_tap, self.b), axis=1) - \
                           np.sum(np.multiply(self.y_tap[:, 1:], self.a[1:]), axis=1)

        data = self.y_tap[:, 0]
        return data

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def reset_tap(self):
        self.x_tap.fill(0)
        self.y_tap.fill(0)


class RealtimeVrms(DataProcessor):
    def __init__(self, fs=250, channel_num=8, interval_ms=250, offset_ms=0):  # interval in ms
        super().__init__(data_processor_type=DataProcessorType.RealtimeVrms)
        self.fs = fs
        self.channel_num = channel_num
        self.interval_ms = interval_ms
        self.offset_ms = offset_ms
        self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
        self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))

    # def init_buffer(self):
    #     self.data_buffer_size = round(self.fs * self.interval_ms * 0.001)
    #     self.data_buffer = np.zeros((self.channel_num, self.data_buffer_size))

    def process_sample(self, data):
        self.data_buffer[:, 1:] = self.data_buffer[:, : -1]
        self.data_buffer[:, 0] = data
        vrms = np.sqrt(1 / self.data_buffer_size * np.sum(np.square(self.data_buffer), axis=1))
        # vrms = np.mean(self.data_buffer, axis=1)
        # print(vrms)
        return vrms

    def reset_tap(self):
        self.data_buffer.fill(0)


class DataProcessorType(Enum):
    RealtimeNotch = 'RealtimeNotch'
    RealtimeButterBandpass = 'RealtimeButterBandpass'
    RealtimeVrms = 'RealtimeVrms'
    RealtimeAvgRef = 'RealtimeAvgRef'
    RealtimeICAEogProxy = 'RealtimeICAEogProxy'


class RealtimeAvgRef(DataProcessor):
    """Average reference applied only to selected EEG picks (matches MNE behavior)."""
    def __init__(self, channel_num: int, picks=None):
        super().__init__(DataProcessorType.RealtimeAvgRef)
        self.channel_num = channel_num
        self.picks = None if picks is None else np.asarray(list(picks), dtype=int)

    def process_buffer(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        if self.picks is None:
            return data - np.mean(data, axis=0, keepdims=True)

        out = data.copy()
        ref = np.mean(out[self.picks, :], axis=0, keepdims=True)
        out[self.picks, :] = out[self.picks, :] - ref
        return out

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


class RealtimeICAEogProxy(DataProcessor):
    def __init__(
            self,
            fs: int,
            channel_num: int,
            eeg_picks,                        # NEW: ICA only on EEG
            n_components: int = 20,
            fit_duration_s: float = 30.0,
            fp_pair=(0, 1),                   # NEW: (Fp1, Fp2) indices in full stream
            f7f8_pair=(2, 3),                 # NEW: (F7, F8) indices in full stream
            blink_z_thresholds=(3.0, 2.75, 2.5, 2.25, 2.0),
            horiz_z_thresholds=(2.5, 2.25, 2.0, 1.75, 1.5),
            random_state: int = 42,
            max_iter: int = 800,
    ):
        super().__init__(DataProcessorType.RealtimeICAEogProxy)
        if FastICA is None:
            raise ImportError("RealtimeICAEogProxy requires scikit-learn (sklearn).")

        self.fs = int(fs)
        self.channel_num = int(channel_num)

        self.eeg_picks = np.asarray(list(eeg_picks), dtype=int)
        self.eeg_ch = int(len(self.eeg_picks))
        self.n_components = int(min(n_components, self.eeg_ch))
        self.fit_samples = int(round(float(fit_duration_s) * float(fs)))

        self.fp1_idx, self.fp2_idx = map(int, fp_pair)
        self.f7_idx, self.f8_idx = map(int, f7f8_pair)

        self.blink_z_thresholds = tuple(float(x) for x in blink_z_thresholds)
        self.horiz_z_thresholds = tuple(float(x) for x in horiz_z_thresholds)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)

        self._calib = np.zeros((self.eeg_ch, self.fit_samples), dtype=np.float32)
        self._calib_n = 0
        self._ica = None
        self.exclude = []
        self._fitted = False

    def reset_tap(self):
        self._calib.fill(0)
        self._calib_n = 0
        self._ica = None
        self.exclude = []
        self._fitted = False

    def _append_calib(self, x_eeg: np.ndarray):
        n = int(x_eeg.shape[1])
        if n <= 0:
            return
        if n >= self.fit_samples:
            self._calib[:, :] = x_eeg[:, -self.fit_samples:]
            self._calib_n = self.fit_samples
            return
        self._calib = np.roll(self._calib, -n, axis=1)
        self._calib[:, -n:] = x_eeg
        self._calib_n = min(self.fit_samples, self._calib_n + n)

    def _fit(self):
        X = self._calib.T  # (T, C_eeg)
        ica = FastICA(
            n_components=self.n_components,
            random_state=self.random_state,
            max_iter=self.max_iter,
            whiten="unit-variance",
        )
        S = ica.fit_transform(X)  # (T, K)

        # Build proxies from calibration data (FULL-stream indices mapped into EEG space)
        # We need the EEG-calib versions of those channels:
        def _eeg_rel(full_idx: int) -> int:
            # find where full_idx sits inside eeg_picks
            hits = np.where(self.eeg_picks == full_idx)[0]
            if hits.size == 0:
                raise ValueError(f"Channel idx {full_idx} not in eeg_picks")
            return int(hits[0])

        fp1 = self._calib[_eeg_rel(self.fp1_idx), :].astype(np.float64)
        fp2 = self._calib[_eeg_rel(self.fp2_idx), :].astype(np.float64)
        f7  = self._calib[_eeg_rel(self.f7_idx),  :].astype(np.float64)
        f8  = self._calib[_eeg_rel(self.f8_idx),  :].astype(np.float64)

        frontal_ref = 0.5 * (fp1 + fp2)            # matches your offline
        hL = f7 - frontal_ref                      # F7 - frontal_ref
        hR = f8 - frontal_ref                      # F8 - frontal_ref

        K = S.shape[1]
        corr_fp1 = np.zeros((K,), dtype=np.float64)
        corr_fp2 = np.zeros((K,), dtype=np.float64)
        corr_hL  = np.zeros((K,), dtype=np.float64)
        corr_hR  = np.zeros((K,), dtype=np.float64)

        for k in range(K):
            sk = S[:, k]
            corr_fp1[k] = abs(_safe_corr(sk, fp1))
            corr_fp2[k] = abs(_safe_corr(sk, fp2))
            corr_hL[k]  = abs(_safe_corr(sk, hL))
            corr_hR[k]  = abs(_safe_corr(sk, hR))

        def _z(c):
            return (c - c.mean()) / (c.std() + 1e-12)

        # mimic your offline: pick *all* comps that pass threshold (union) and break on first threshold that yields any
        blink_ex = []
        for thr in self.blink_z_thresholds:
            cand = set(np.where(_z(corr_fp1) >= thr)[0].tolist()) | set(np.where(_z(corr_fp2) >= thr)[0].tolist())
            if len(cand) > 0:
                blink_ex = sorted(cand)
                break

        horiz_ex = []
        for thr in self.horiz_z_thresholds:
            cand = set(np.where(_z(corr_hL) >= thr)[0].tolist()) | set(np.where(_z(corr_hR) >= thr)[0].tolist())
            if len(cand) > 0:
                horiz_ex = sorted(cand)
                break

        self.exclude = sorted(set(blink_ex + horiz_ex))
        self._ica = ica
        self._fitted = True

        GREEN = "\033[92m"
        RESET = "\033[0m"

        print(
            f"{GREEN}[RealtimeICAEogProxy] fitted: exclude={self.exclude} "
            f"(blink max={max(corr_fp1.max(), corr_fp2.max()):.3f}, "
            f"horiz max={max(corr_hL.max(), corr_hR.max()):.3f}){RESET}"
        )

    def process_buffer(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float32)
        x_eeg = data[self.eeg_picks, :]

        if not self._fitted:
            self._append_calib(x_eeg)
            if self._calib_n >= self.fit_samples:
                self._fit()   # (still blocking; if you want I’ll give you the threaded version)
            return data

        if self._ica is None or len(self.exclude) == 0:
            return data

        X = x_eeg.T
        S = self._ica.transform(X)
        S[:, self.exclude] = 0.0
        Xc = self._ica.inverse_transform(S).T

        out = data.copy()
        out[self.eeg_picks, :] = np.asarray(Xc, dtype=np.float32)
        return out


def get_processor_class(data_processor_type):
    if data_processor_type == DataProcessorType.RealtimeNotch:
        return RealtimeNotch
    elif data_processor_type == DataProcessorType.RealtimeButterBandpass:
        return RealtimeButterBandpass
    elif data_processor_type == DataProcessorType.RealtimeVrms:
        return RealtimeVrms
    elif data_processor_type == DataProcessorType.RealtimeAvgRef:
        return RealtimeAvgRef
    elif data_processor_type == DataProcessorType.RealtimeICAEogProxy:
        return RealtimeICAEogProxy


if __name__ == '__main__':
    a = RealtimeButterBandpass()
    print(type)