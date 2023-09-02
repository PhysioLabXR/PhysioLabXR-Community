import time

from physiolabxr.scripting.RenaScript import RenaScript


class LongLoop(RenaScript):
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
        for i in range(int(1e20)):
            print('Running a very long loop ' + str(i))

    def cleanup(self):
        print('Cleanup function is called')
