from physiolabxr.examples.physio_helpers import load_model
from physiolabxr.scripting.RenaScript import RenaScript


class MultiModelTargetPrediction(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')
        self.p300_model = load_model(self.params["model_path"])

    # loop is called <Run Frequency> times per second
    def loop(self):
        if self.inputs["fixations"][-1]:
            y_pred = self.p300_model.predict(self.inputs["fMRI"],
                                             self.inputs["Eyelink 1000"]["Pupil Diameter"], pupil_resample_rate=200)
            self.outputs["p300Detect"] = y_pred

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
