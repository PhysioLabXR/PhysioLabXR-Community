from physiolabxr.examples.physio_helpers import detect_fixations
from physiolabxr.scripting.RenaScript import RenaScript


class FixationDetection(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')

    # loop is called <Run Frequency> times per second
    def loop(self):
        fixations = detect_fixations(self.inputs["Eyelink 1000"]["Gaze Vector"], resample_rate=200)
        self.outputs["fixations"] = fixations

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
