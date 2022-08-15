import time

import numpy as np

from rena.scripting.RenaScript import RenaScript


class IndexPen(RenaScript):
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
        print('Received input of shape ' + str(self.inputs['Dummy-8Chan'].shape))
        self.outputs['SampleOutput'] = np.random.rand(10)

    def cleanup(self):
        print('Cleanup function is called')
