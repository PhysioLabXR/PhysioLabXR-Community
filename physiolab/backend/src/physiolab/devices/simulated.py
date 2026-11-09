import numpy as np

from physiolab.contracts import Channel, SampleBatch, StreamDescriptor


class SimulatedEMG:
    """Deterministic 20 Hz signal alternating rest/activation every two seconds."""

    descriptor = StreamDescriptor(
        id="emg",
        device_id="simulated-emg",
        modality="emg",
        sample_rate_hz=100,
        channels=[Channel(name="EMG", unit="arbitrary")],
        delivery="simulated",
    )

    def __init__(self, start_time: float):
        self.start_time = start_time
        self.sequence = 0
        self.running = False

    def start(self):
        self.running = True

    def read(self) -> SampleBatch:
        if not self.running:
            raise RuntimeError("Connector is stopped")
        t = (np.arange(20) + self.sequence * 20) / self.descriptor.sample_rate_hz
        amplitude = np.where((t // 2).astype(int) % 2 == 0, 0.05, 0.8)
        values = amplitude * np.sin(2 * np.pi * 20 * t)
        batch = SampleBatch(
            stream=self.descriptor,
            timestamps=(self.start_time + t).tolist(),
            samples=values[:, None].tolist(),
            sequence=self.sequence,
        )
        self.sequence += 1
        return batch

    def stop(self):
        self.running = False
