######################################################################################
# Connects to Gazepoint API server and feeds eye-tracking data to a shared queue
# Created on 2023-10-29 by Zoe Zhou
######################################################################################

import socket
from threading import Thread
from queue import Queue

HOST = '127.0.0.1'
PORT = 4242
ADDRESS = (HOST, PORT)

data_queue = Queue()


def data_collector():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(ADDRESS)

    # Send commands to initialize data streaming
    s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))  # fixation

    # diameter
    s.send(str.encode('<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />\r\n'))
    s.send(str.encode('<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />\r\n'))

    s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))

    while True:
        data = s.recv(1024).decode().split(" ")

        keys = ["FPOGX", "FPOGY", "FPOGD", "FPOGID", "LPUPILD", "RPUPILD"]
        type_map = {"FPOGX": float, "FPOGY": float, "FPOGD": float, "FPOGID": int, "LPUPILD": float, "RPUPILD": float}
        result = {key: 0 for key in keys}

        for el in data:
            for key in keys:
                if key in el:
                    result[key] = type_map[key](el.split("\"")[1])

        data_queue.put(result)


collector_thread = Thread(target=data_collector)
collector_thread.start()
