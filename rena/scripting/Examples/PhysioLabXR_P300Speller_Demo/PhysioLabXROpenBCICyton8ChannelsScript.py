import time

import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from rena.scripting.RenaScript import RenaScript


class PhysioLabXROpenBCICyton8ChannelsScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)


    # Start will be called once when the run button is hit.
    def init(self):

        self.serial_port_available = False
        self.impedance_available = False

        if "serial_port" in self.params: # check
            if type(self.params["serial_port"]) is str:
                self.serial_port_available = True
            else:
                print("serial_port should be a string (e.g. COM3)")
        else:
            print("serial_port is not set. Please set it in the parameters tab (e.g. COM3)")

        if "impedance" in self.params: # check
            if type(self.params["impedance"]) is str:
                self.impedance_available = True
            else:
                print("impedance should be a boolean (e.g. True)")
        else:
            print("impedance is not set. Please set it in the parameters tab (e.g. True)")

        print("serial_port_available: ", self.serial_port_available)
        print("impedance_available: ", self.impedance_available)


        time.sleep(1)






    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
