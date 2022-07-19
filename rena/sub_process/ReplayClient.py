from rena import config
import threading

from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.sub_process.server_workers import RenaDSPUnit, DSPServerWorker

class ReplayClient(threading.Thread):
    def __init__(self, replay_command_interface):
        super().__init__()
        self.command_interface = replay_command_interface

    def run(self):
        while True:
            pass

def start_replay_client():

    print("Replay Client Started")
    replay_command_interface = RenaTCPInterface(stream_name=config.replay_server_name,
                                        port_id=config.replay_server_port,
                                        identity='client')
    replay_client_thread = ReplayClient(replay_command_interface)
