"""Haosen owns signal transforms. State is per run, never shared across users."""

import numpy as np
from scipy.signal import butter, sosfilt

from physiolab.contracts import SampleBatch


class Bandpass:
    def __init__(self, low_hz: float, high_hz: float, rate_hz: float):
        if not 0 < low_hz < high_hz < rate_hz / 2:
            raise ValueError("Bandpass requires 0 < low < high < Nyquist")
        self.sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=rate_hz, output="sos")
        self.state = None

    def process(self, batch: SampleBatch) -> SampleBatch:
        samples = np.asarray(batch.samples)
        if self.state is None:
            self.state = np.zeros((len(self.sos), 2, samples.shape[1]))
        filtered, self.state = sosfilt(self.sos, samples, axis=0, zi=self.state)
        return batch.model_copy(update={"samples": filtered.tolist()})


class SlidingWindow:
    def __init__(self, size: int, stride: int):
        if not 1 <= stride <= size <= 1000:
            raise ValueError("Window requires 1 <= stride <= size <= 1000")
        self.size, self.stride = size, stride
        self.samples: list[list[float]] = []
        self.timestamps: list[float] = []

    def process(self, batch: SampleBatch) -> list[SampleBatch]:
        self.samples.extend(batch.samples)
        self.timestamps.extend(batch.timestamps)
        windows = []
        while len(self.samples) >= self.size:
            windows.append(
                batch.model_copy(
                    update={
                        "samples": self.samples[: self.size],
                        "timestamps": self.timestamps[: self.size],
                    }
                )
            )
            del self.samples[: self.stride]
            del self.timestamps[: self.stride]
        return windows
