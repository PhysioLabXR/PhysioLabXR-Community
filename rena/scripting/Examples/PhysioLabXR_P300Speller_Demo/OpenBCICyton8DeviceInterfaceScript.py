import numpy as np
import brainflow
from rena.scripting.RenaScript import RenaScript

class OpenBCICyton8DeviceInterfaceScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # test network

    # Start will be called once when the run button is hit.
    def init(self):
        print('Init function is called')
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
