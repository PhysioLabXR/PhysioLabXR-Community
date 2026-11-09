import numpy as np

from physiolab.contracts import Inference, SampleBatch


class DemoEMGModel:
    """Fixture classifier, not a trained physiological/foundation model or focus estimate."""

    id = "demo-emg-rms"
    promptable = False

    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold

    def infer(self, window: SampleBatch) -> Inference:
        rms = float(np.sqrt(np.mean(np.asarray(window.samples) ** 2)))
        return Inference(
            timestamp=window.timestamps[-1],
            model_id=self.id,
            label="gesture" if rms >= self.threshold else "rest",
            value=rms,
            unit="rms",
        )
