from physiolabxr.rpc.decorator import rpc, async_rpc
from physiolabxr.scripting.RenaScript import RenaScript

class AsyncRPCExample(RenaScript):
    def __init__(self, *args, **kwargs):
        """Please do not edit this function"""
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        print(f'Loop: async server')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    @async_rpc
    def AsyncOneArgOneReturn(self, input0: str) -> str:
        """
        This is an example of a RPC method that takes one argument and returns one value.

        It is an async rpc method. The input protobuf is called AsyncOneArgOneReturnRequest,

        The input protobuf is called ExampleOneArgOneReturnRequest,
        and the output is called ExampleOneArgOneReturnResponse
        Args:
            input0 (str): the input
        """
        cycle = 100000
        for i in range(cycle):
            print(f"Spinning {i}/{cycle}")
        return f"received: {input0}"