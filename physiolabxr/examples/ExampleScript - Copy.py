import time

import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript


class ExampleScript(RenaScript):
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
        self.outputs['output1'] = np.random.rand(10)
        self.outputs['output2'] = np.array([self.params['param2']])
        print('Received input of shape ' + str(self.inputs['Dummy-8Chan'].shape))
        print('Param value 1 is {}'.format(self.params['param1']))

    def cleanup(self):
        print('Cleanup function is called')
