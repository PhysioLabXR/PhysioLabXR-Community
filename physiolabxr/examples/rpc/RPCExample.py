import numpy as np

from physiolabxr.rpc.decorator import rpc
from physiolabxr.scripting.RenaScript import RenaScript


class RPCExample(RenaScript):
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
        print(f'Loop: rpc server {self.rpc_server}')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    @rpc
    def PredictFromInput(self, arg0: str, arg1) -> str:
        """
        it is conventional to use camal case for RPC methods
        """
        # return int(self.inputs["stream"][0][0, 0])
        return "Hello from RPCExample! received input: " + arg0