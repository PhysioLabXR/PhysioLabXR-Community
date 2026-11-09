"""Wrap an existing instance without importing Qt or native SDKs into the web service.

Run hardware adapters in a local bridge process; transmit normalized SampleBatch to the
future ingest endpoint. The older interface returns CHANNELS x SAMPLES and may use a
monotonic/device clock. Supply an explicit clock converter; never pretend it is Unix time.
"""

from collections.abc import Callable
from typing import Any

import numpy as np

from physiolab.contracts import SampleBatch, StreamDescriptor


class LegacyDeviceAdapter:
    def __init__(
        self, device: Any, descriptor: StreamDescriptor, to_unix_seconds: Callable[[float], float]
    ):
        self.device, self.descriptor = device, descriptor
        self.to_unix_seconds = to_unix_seconds
        self.sequence = 0

    def start(self):
        self.device.start_stream()

    def read(self) -> SampleBatch | None:
        result = self.device.process_frames()
        frames, timestamps = result[:2]  # DeviceInterface: 3 values; LSLInletInterface: 2
        if not len(timestamps):
            return None
        batch = SampleBatch(
            stream=self.descriptor,
            timestamps=[self.to_unix_seconds(float(t)) for t in timestamps],
            samples=np.asarray(frames).T.tolist(),
            sequence=self.sequence,
        )
        self.sequence += 1
        return batch

    def stop(self):
        self.device.stop_stream()
