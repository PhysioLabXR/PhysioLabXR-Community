import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from rena.scripting.RenaScript import RenaScript


class PhysioLabXROpenBCICyton8ChannelsScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.port = None
        self.impedance = None
        self.params_available = False
        self.board_connected = False

    # Start will be called once when the run button is hit.
    def init(self):
        # self.params is a dictionary of parameters

        assert 1 == 2, 'Please implement this function'

        try:
            self.port = self.params['port']
            self.params_available = False
        except KeyError:
            print('parameter port is not set')
            return

        try:
            self.impedance = self.params['impedance']
            self.params_available = False
        except KeyError:
            print('parameter impedance is not set')
            return

        self.board_params = BrainFlowInputParams()

        raise NotImplementedError('Please implement this function')
        print("John's code starts here")



    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')



    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
