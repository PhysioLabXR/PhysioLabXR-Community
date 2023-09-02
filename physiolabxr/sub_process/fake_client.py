from physiolabxr.sub_process.TCPInterface import *

rena_tcp_request_object = RenaTCPAddDSPWorkerRequestObject(stream_name='mmWave',
                                                           port_id=1234,
                                                           identity='server',
                                                           processor_dict={})
rena_main_client = RenaTCPInterface(stream_name='Main Server', port_id=999999, identity='client')
rena_main_client.send_obj(rena_tcp_request_object)
