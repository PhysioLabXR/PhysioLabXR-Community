from typing import Protocol

from physiolab.contracts import Inference, SampleBatch


class PhysiologicalModel(Protocol):
    """Haosen: add model cards, input compatibility and provenance with each implementation."""

    id: str
    promptable: bool

    def infer(self, window: SampleBatch) -> Inference: ...
