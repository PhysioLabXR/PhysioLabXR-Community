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
        # print(f'Loop: rpc server')
        pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    @rpc
    def ExampleOneArgOneReturn(self, input0: str) -> str:
        """
        This is an example of a RPC method that takes one argument and returns one value

        The input protobuf is called ExampleOneArgOneReturnRequest,
        and the output is called ExampleOneArgOneReturnResponse
        Args:
            input0 (str): the input
        """
        return f"Miaomiaomiao: {input0}"

    @rpc
    def TestRPCTwoArgTwoReturn(self, input0: str, input1: int) -> (str, int):
        """
        This is an example of a RPC method that takes two arguments and returns two values

        The input protobuf is called TestRPCTwoArgTwoReturnRequest,
        and the output is called TestRPCTwoArgTwoReturnResponse
        Args:
            input0 (str): the first input
            input1 (int): the second input

        Returns:
            TestRPCTwoArgTwoReturnResponse: the output
        """
        return f"received {input0}", input1

    @rpc
    def TestRPCNoInputNoReturn(self):
        """
        # this rpc does not return nor does it take any input
        """
        return "No input no return RPC called"

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
        return "No Args"
