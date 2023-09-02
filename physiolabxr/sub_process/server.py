from physiolabxr.configs import config
from physiolabxr.sub_process.TCPInterface import RenaTCPInterface
from physiolabxr.sub_process.server_workers import DSPServerWorker

if __name__ == '__main__':
    print("Server Started")
    dsp_server_workers = {}
    # init main server

    rena_main_server = RenaTCPInterface(stream_name=config.rena_server_name,
                                        port_id=config.rena_server_port,
                                        identity='server')

    while True:
        print('receiving request')
        rena_tcp_request_object = rena_main_server.recv_obj()  # send the object back for finishing confirmation
        rena_tcp_server_interface = RenaTCPInterface(stream_name=rena_tcp_request_object.stream_name,
                                                     port_id=rena_tcp_request_object.port_id,
                                                     identity=rena_tcp_request_object.identity)

        dsp_server_workers[rena_tcp_request_object.stream_name] = DSPServerWorker(
            RenaTCPInterface=rena_tcp_server_interface)
        dsp_server_workers[rena_tcp_request_object.stream_name].start()

        rena_main_server.send_obj(rena_tcp_request_object) # send the object back for finishing confirmation
