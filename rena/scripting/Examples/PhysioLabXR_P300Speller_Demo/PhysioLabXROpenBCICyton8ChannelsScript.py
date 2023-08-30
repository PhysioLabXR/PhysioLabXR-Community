import time
import brainflow
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from pylsl import StreamOutlet, StreamInfo
import pylsl
from rena.scripting.RenaScript import RenaScript


class PhysioLabXROpenBCICyton8ChannelsScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)


    # Start will be called once when the run button is hit.
    def init(self):
        # check if the parameters are set

        if "serial_port" not in self.params: # check
            while True:
                print("serial_port is not set. Please set it in the parameters tab (e.g. COM3)")
                time.sleep(1)
        else:
            if type(self.params["serial_port"]) is not str:
                while True:
                    print("serial_port should be a string (e.g. COM3)")
                    time.sleep(1)

        # if "impedance" not in self.params: # check
        #     while True:
        #         print("impedance is not set. Please set it in the parameters tab (e.g. True)")
        #         time.sleep(1)
        # else:
        #     if type(self.params["impedance"]) is not bool:
        #         while True:
        #             print("impedance should be a boolean (e.g. True)")
        #             time.sleep(1)


        print("serial_port: ", self.params["serial_port"])
        # print("impedance: ", self.params["impedance"])

        # try init board
        self.brinflow_input_params = BrainFlowInputParams()

        # assign serial port from params to brainflow input params
        self.brinflow_input_params.serial_port = self.params["serial_port"]

        self.brinflow_input_params.ip_port = 0
        self.brinflow_input_params.mac_address = ''
        self.brinflow_input_params.other_info = ''
        self.brinflow_input_params.serial_number = ''
        self.brinflow_input_params.ip_address = ''
        self.brinflow_input_params.ip_protocol = 0
        self.brinflow_input_params.timeout = 0
        self.brinflow_input_params.file = ''

        # set board id to Cyton 8-channel (0)
        self.board_id = 0 # Cyton 8-channel

        try:
            self.board = BoardShim(self.board_id, self.brinflow_input_params)
            self.board.prepare_session()
            self.board.start_stream(45000, '') # 45000 is the default ring buffer size
            print("OpenBCI Cyton 8 Channels. Sensor Start.")
        except brainflow.board_shim.BrainFlowError:
            while True:
                print('Board is not ready. Start Fild. Please check the serial port and try again.')
                time.sleep(1)

        # init lsl outlet
        self.outlet_info = StreamInfo(name="OpenBCICyton8Channels", type="EEG", channel_count=8,
                                   nominal_srate=250, channel_format='float32',
                                   source_id='Cyton8Channels')
        self.stream_outlet = StreamOutlet(self.outlet_info)



    # loop is called <Run Frequency> times per second
    def loop(self):
        timestamp_channel = self.board.get_timestamp_channel(0)
        eeg_channels = self.board.get_eeg_channels(0)
        # print(timestamp_channel)
        # print(eeg_channels)

        data = self.board.get_board_data()

        timestamps = data[timestamp_channel]
        data = data[eeg_channels]

        absolute_time_to_lsl_time_offset = time.time() - pylsl.local_clock()

        # send data to lsl
        for timestamp, frame in zip(timestamps, data.T):
            self.stream_outlet.push_sample(frame, timestamp-absolute_time_to_lsl_time_offset)


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Stop OpenBCI Cyton 8 Channels. Sensor Stop.')
        # try:
        #     self.board.stop_stream()
        #     print('OpenBCIInterface: stopped streaming.')
        #     self.board.release_session()
        #     print('OpenBCIInterface: released session.')
        # except brainflow.board_shim.BrainFlowError as e:
        #     print(e)
