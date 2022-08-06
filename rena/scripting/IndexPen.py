import time

from rena.scripting.RenaScript import RenaScript


class IndexPen(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def start(self):
        print('Start function is called')

    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')
        time.sleep(1)