import threading
from abc import ABC, abstractmethod

class RenaScript(ABC, threading.Thread):
    """
    An abstract class for implementing scripting models.
    """
    def __init__(self, inputs, outputs, params, port):
        """

        :param inputs:
        :param outputs:
        :param params:
        :param port: the port to which we bind the
        """
        super().__init__()
        # create
        input_streams = None
        expected_preprocessed_input_size = None
        # self.command_info_interface = command_info_interface

        # create data buffers


    @abstractmethod
    def start(self):
        """
        Start will be called once when the run button is hit.
        """
        pass

    @abstractmethod
    def Loop(self):
        """
        Loop is called
        """
        pass

    def run(self):
        self.start()
        # start the loop here, accept interrupt command

def start_replay_server():
    print("Replay Client Started")
    command_info_interface = RenaTCPInterface(stream_name='RENA_REPLAY',
                                              port_id=config.replay_port,
                                              identity='server',
                                              pattern='router-dealer')

    replay_client_thread = ReplayServer(command_info_interface)
    replay_client_thread.start()

if __name__ == '__main__':
    start_replay_server()