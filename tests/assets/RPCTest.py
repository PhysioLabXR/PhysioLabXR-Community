from typing import Union

import numpy as np

from physiolabxr.rpc.decorator import rpc
from physiolabxr.scripting.RenaScript import RenaScript


class RPCTest(RenaScript):
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
    def TestRPCOneArgOneReturn(self, input0: str) -> str:
        """
        it is conventional to use camal case for RPC methods
        """
        # return int(self.inputs["stream"][0][0, 0])
        return "Hello from RPCExample! received input: " + input0

    @rpc
    def TestRPCTwoArgTwoReturn(self, input0: str, input1: int) -> Union[str, int]:
        """
        it is conventional to use camal case for RPC methods
        """
        # return int(self.inputs["stream"][0][0, 0])
        return "Hello from RPCExample! received input: " + input0

    @rpc
    def TestRPCNoInputNoReturn(self):
        """
        # this rpc does not return nor does it take any input
        """
        # return int(self.inputs["stream"][0][0, 0])
        return "Hello from RPCExample! received input: "

    @rpc
    def TestRPCNoReturn(self, input0: float):
        """
        # this rpc does not return anything
        """
        # return int(self.inputs["stream"][0][0, 0])
        return f"Hello from RPCExample! received input: {input0}"

    @rpc
    def TestRPCNoArgs(self) -> str:
        """
        # this rpc does not return anything
        """
        # return int(self.inputs["stream"][0][0, 0])
        return "Hello from RPCExample! received input: "
