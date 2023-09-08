import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript


class BaseRenaScript(RenaScript):
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
        print('Loop function is called')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
