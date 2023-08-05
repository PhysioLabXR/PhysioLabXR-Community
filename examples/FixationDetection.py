import numpy as np

from rena.scripting.RenaScript import RenaScript
from rena.scripting.physio.epochs import get_event_locked_data, buffer_event_locked_data, get_baselined_event_locked_data


class FixationDetection(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        self.outputs['FixationOverlay'] = np.random.randint(0, 255, [1, 400*400*3], dtype=np.uint8)
        print("sent to output")


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')


