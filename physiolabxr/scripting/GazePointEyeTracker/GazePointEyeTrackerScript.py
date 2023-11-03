# This is an example script for PhysioLabXR. It is a simple script that reads data from OpenBCI Cyton 8 Channels and sends it to Lab Streaming Layer.
# The output stream name is "OpenBCICyton8Channels"


import time
import brainflow
import pylsl
from physiolabxr.scripting.RenaScript import RenaScript
import socket
from queue import Queue


class PhysioLabXROpenBCICyton8ChannelsScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)


    # Start will be called once when the run button is hit.
    def init(self):
        self.HOST = '127.0.0.1'
        self.PORT = 4242
        self.ADDRESS = (self.HOST, self.PORT)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(self.ADDRESS)

        # Send commands to initialize data streaming
        self.s.send(str.encode('<SET ID="ENABLE_SEND_POG_FIX" STATE="1" />\r\n'))  # fixation

        # diameter
        self.s.send(str.encode('<SET ID="ENABLE_SEND_EYE_LEFT" STATE="1" />\r\n'))
        self.s.send(str.encode('<SET ID="ENABLE_SEND_EYE_RIGHT" STATE="1" />\r\n'))

        self.s.send(str.encode('<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'))

        pass
    # loop is called <Run Frequency> times per second
    def loop(self):
        data = self.s.recv(1024).decode().split(" ")
        keys = ["FPOGX", "FPOGY", "FPOGD", "FPOGID", "LPUPILD", "RPUPILD"]
        type_map = {"FPOGX": float, "FPOGY": float, "FPOGD": float, "FPOGID": int, "LPUPILD": float, "RPUPILD": float}
        result = {key: 0 for key in keys}

        for el in data:
            for key in keys:
                if key in el:
                    result[key] = type_map[key](el.split("\"")[1])

        output_data = list(result.values())
        # convert output_data to a list of floats
        output_data = [float(i) for i in output_data]

        timestamp = pylsl.local_clock()

        self.set_output(stream_name="GazePoint", data=output_data, timestamp=timestamp)


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Stop OpenBCI Cyton 8 Channels. Sensor Stop.')


