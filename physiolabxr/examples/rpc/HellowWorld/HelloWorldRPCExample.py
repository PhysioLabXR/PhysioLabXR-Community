from physiolabxr.rpc.decorator import rpc
from physiolabxr.scripting.RenaScript import RenaScript


class HelloWorldRPC(RenaScript):
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
        print(f'Loop: rpc server')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    @rpc
    def SayHello(self, name: str) -> str:
        """
        it is conventional to use camal case for RPC methods
        """
        return "Hello, %s!" % name