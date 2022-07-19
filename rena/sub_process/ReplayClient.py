from rena import config
from rena.sub_process.TCPInterface import RenaTCPInterface
from rena.sub_process.server_workers import RenaDSPUnit, DSPServerWorker

if __name__ == '__main__':
    print("Replay Client Started")
    replay_interface = RenaTCPInterface(stream_name=config.replay_server_name,
                                        port_id=config.replay_server_port,
                                        identity='client')
