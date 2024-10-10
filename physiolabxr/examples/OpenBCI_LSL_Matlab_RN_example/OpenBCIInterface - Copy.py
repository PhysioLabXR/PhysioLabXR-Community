import time

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet
from physiolabxr.utils.realtime_DSP import RealtimeNotch, RealtimeButterBandpass, RealtimeVrms


class OpenBCIInterface:

    def __init__(self, serial_port='COM6', board_id=0, log='store_true', streamer_params='',
                 ring_buffer_size=45000):  # default board_id 2 for Cyton
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
        self.board_id = board_id
        self.info_eeg = None
        self.info_aux = None
        self.outlet_eeg = None
        self.outlet_aux = None

        if (log):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        self.board = BoardShim(board_id, params)

    def start_sensor(self):
        # tell the sensor to start sending frames
        self.board.prepare_session()
        print('OpenBCIInterface: connected to sensor')
        print(self.board.get_board_id())

        try:
            self.board.start_stream(self.ring_buffer_size, self.streamer_params)
            self.infor_test()
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

    def create_lsl(self, name='OpenBCI_Cython_8_LSL', type='EEG', channel_count=8,
                   nominal_srate=250.0, channel_format='float32',
                   source_id='Cyton_0'):

        self.info_eeg = StreamInfo(name=name, type=type, channel_count=channel_count,
                                   nominal_srate=nominal_srate, channel_format=channel_format,
                                   source_id='')

        chns = self.info_eeg.desc().append_child('channels')

        self.labels = ['Fp1', 'Fp2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2']

        for label in self.labels:
            ch = chns.append_child("channel")
            ch.append_child_value('label', label)
            ch.append_child_value('unit', 'microvolts')
            ch.append_child_value('type', 'EEG')

        self.info_eeg.desc().append_child_value('manufacturer', 'OpenBCI Inc.')
        self.outlet_eeg = StreamOutlet(self.info_eeg)

        print("--------------------------------------\n" + \
              "LSL Configuration: \n" + \
              "  Stream 1: \n" + \
              "      Name: " + name + " \n" + \
              "      Type: " + type + " \n" + \
              "      Channel Count: " + str(channel_count) + "\n" + \
              "      Sampling Rate: " + str(nominal_srate) + "\n" + \
              "      Channel Format: " + channel_format + " \n" + \
              "      Source Id: " + source_id + " \n")

    def push_sample(self, samples):
        self.outlet_eeg.push_sample(samples)

    def infor_test(self):
        print(self.board.get_eeg_names(self.board_id))
        print(self.board.get_sampling_rate(self.board_id))
        print(self.board.get_board_id())
        print(self.board.get_package_num_channel(self.board_id))
        print(self.board.get_timestamp_channel(self.board_id))
        print(self.board.get_eeg_channels(self.board_id))
        print(self.board.get_accel_channels(self.board_id))
        print(self.board.get_marker_channel(self.board_id))
        print(self.board.get_other_channels(self.board_id))
        print(self.board.get_analog_channels(self.board_id))
        print(self.board.get_other_channels(self.board_id))


# def run_test():
#     data = np.empty(shape=(24, 0))
#     print('Started streaming')
#     start_time = time.time()
#     while 1:
#         try:
#             new_data = openBCI_interface.process_frames()
#             data = np.concatenate((data, new_data), axis=-1)  # get all data and remove it from internal buffer
#         except KeyboardInterrupt:
#             f_sample = data.shape[-1] / (time.time() - start_time)
#             print('Stopped streaming, sampling rate = ' + str(f_sample))
#             break
#     return data

def run_test():
    # data = np.empty(shape=(24, 0))
    print('Started streaming')
    start_time = time.time()
    notch = RealtimeNotch(w0=60, Q=25, fs=250, channel_num=8)
    butter_bandpass = RealtimeButterBandpass(lowcut=1, highcut=50, fs=250, order=6, channel_num=8)
    # vrms_converter = RealtimeVrms(fs=250, channel_num=8, interval_ms=500, offset_ms=0)

    # starting time
    start_time = time.time()
    while 1:
        try:
            new_data = openBCI_interface.process_frames()
            # data = np.concatenate((data, new_data), axis=-1)  # get all data and remove it from internal buffer
            for data in new_data.T:
                eeg_data = data[1:9]
                print(eeg_data)
                # aux_data = data[9:12]
                # ######### notch and butter

                # eeg_data = notch.process_sample(eeg_data)
                # eeg_data = butter_bandpass.process_sample(eeg_data)

                # eeg_data = vrms_converter.process_data(eeg_data)
                # push sample to lsl with interval
                # if time.time() - start_time > 0.35:
                openBCI_interface.push_sample(samples=eeg_data)
                # print(eeg_data)
                    # print(eeg_data)
                    # start_time = time.time()
        except KeyboardInterrupt:
            # f_sample = data.shape[-1] / (time.time() - start_time)
            print('Stopped streaming, sampling rate = ' + str())
            break
    return True


if __name__ == "__main__":
    openBCI_interface = OpenBCIInterface(serial_port='COM5')
    openBCI_interface.create_lsl()
    openBCI_interface.start_sensor()
    data = run_test()
    openBCI_interface.stop_sensor()
