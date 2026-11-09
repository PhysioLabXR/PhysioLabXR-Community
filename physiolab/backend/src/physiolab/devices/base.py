"""Serena owns device discovery, permissions, lifecycle and normalized sample delivery."""

from typing import Protocol

from physiolab.contracts import SampleBatch, StreamDescriptor


class DeviceConnector(Protocol):
    descriptor: StreamDescriptor

    def start(self) -> None: ...
    def read(self) -> SampleBatch: ...
    def stop(self) -> None: ...
