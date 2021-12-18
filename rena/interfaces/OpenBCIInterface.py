import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations



class OpenBCIInterface:

    def __init__(self, serial_port='COM6', board_id=2, log='store_true', streamer_params='', ring_buffer_size=45000):  # default board_id 2 for Cyton
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.ip_port = 0
        params.mac_address = ''
        params.other_info = ''
        params.serial_number = ''
        params.ip_address = ''
        params.ip_protocol = 0
        params.timeout = 0
        params.file = ''
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size

        if (log):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        self.board = BoardShim(board_id, params)

    def start_sensor(self):
        # tell the sensor to start sending frames
        self.board.prepare_session()
        print('OpenBCIInterface: connected to sensor')
        try:
            self.board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError:
            print('OpenBCIInterface: Board is not ready.')

    def process_frames(self):
        # return one or more frames of the sensor
        frames = self.board.get_board_data()
        return frames

    def stop_sensor(self):
        try:
            self.board.stop_stream()
            print('OpenBCIInterface: stopped streaming.')
            self.board.release_session()
            print('OpenBCIInterface: released session.')
        except brainflow.board_shim.BrainFlowError as e:
            print(e)


def run_test():
    data = np.empty(shape=(31, 0))
    print('Started streaming')
    start_time = time.time()
    while 1:
        try:
            new_data = openBCI_interface.process_frames()
            data = np.concatenate((data, new_data), axis=-1)  # get all data and remove it from internal buffer
        except KeyboardInterrupt:
            f_sample = data.shape[-1] / (time.time() - start_time)
            print('Stopped streaming, sampling rate = ' + str(f_sample))
            break
    return data


if __name__ == "__main__":
    openBCI_interface = OpenBCIInterface()
    openBCI_interface.start_sensor()
    data = run_test()
    openBCI_interface.stop_sensor()
