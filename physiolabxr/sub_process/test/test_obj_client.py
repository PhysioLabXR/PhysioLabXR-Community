import time

import numpy as np

from physiolabxr.sub_process.TCPInterface import RenaTCPInterface, RenaTCPClient, RenaTCPObject
from physiolabxr.sub_process.processor import dsp_processor


dsp_client_interface = RenaTCPInterface(stream_name='John',
                                        port_id=111,
                                        identity='client')
dsp_client = RenaTCPClient(RENATCPInterface=dsp_client_interface)

while True:
    dsp_client.process_data(RenaTCPObject(data=np.array([1, 2, 3, 4])))
    # time.sleep(1)

