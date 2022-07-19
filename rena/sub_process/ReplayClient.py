from rena import config
import threading

from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.sub_process.server_workers import RenaDSPUnit, DSPServerWorker


class ReplayClient(threading.Thread):
    def __init__(self, receive_command_interface):
        super().__init__()
        self.receive_command_interface = receive_command_interface
        self.is_replaying = False

    def run(self):
        while True:
            if not self.is_replaying:
                print('ReplayClient: not replaying, pending on start replay command')
                self.receive_command_interface.socket.recv()
                print('ReplayClient: started replaying')
            else:
                a = self.receive_command_interface.poller.poll(timeout=1)
                pass


def start_replay_client():
    print("Replay Client Started")
    receive_command_interface = RenaTCPInterface(stream_name='RENA_REPLAY_CLIENT',
                                                 port_id=config.replay_port,
                                                 identity='client',
                                                 pattern='pipeline')
    replay_client_thread = ReplayClient(receive_command_interface)
    replay_client_thread.start()
