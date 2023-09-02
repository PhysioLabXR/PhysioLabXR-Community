import time

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from pylsl import StreamInfo, StreamOutlet
from physiolabxr.utils.realtime_DSP import RealtimeNotch, RealtimeButterBandpass, RealtimeVrms


class OpenBCIDeviceInterface:

    def __init__(self, stream_name, stream_type='EEG', serial_port='COM5', board_id="0",
                 log='store_true', streamer_params='',
                 ring_buffer_size=45000):  # default board_id 2 for Cyton
        self.params = BrainFlowInputParams()
        self.params.serial_port = serial_port
        self.params.ip_port = 0
        self.params.mac_address = ''
        self.params.other_info = ''
        self.params.serial_number = ''
        self.params.ip_address = ''
        self.params.ip_protocol = 0
        self.params.timeout = 0
        self.params.file = ''

        self.stream_name = stream_name
        self.stream_type = stream_type
        self.board_id = int(board_id)
        self.streamer_params = streamer_params
        self.ring_buffer_size = ring_buffer_size

        if (log):
            BoardShim.enable_dev_board_logger()
        else:
            BoardShim.disable_board_logger()

        try:
            self._board = BoardShim(self.board_id, self.params)
            self.info_print()
        except brainflow.board_shim.BrainFlowError:
            print('Cannot connect to board')

    def start_sensor(self):
        # tell the sensor to start sending frames
        try:
            self._board.prepare_session()
        except brainflow.board_shim.BrainFlowError:
            raise AssertionError('Unable to connect to device: please check the sensor connection or the COM port number in device preset.')
        print('OpenBCIInterface: connected to sensor')

        try:
            self._board.start_stream(self.ring_buffer_size, self.streamer_params)
        except brainflow.board_shim.BrainFlowError:
            raise AssertionError('Unable to connect to device: please check the sensor connection or the COM port number in device preset.')
        print('OpenBCIInterface: connected to sensor')

        # self.create_lsl(name=self.stream_name,
        #                 type=self.stream_type,
        #                 nominal_srate=self.board.get_sampling_rate(self.board_id),
        #                 channel_format='float32',
        #                 source_id='Cyton_' + str(self.board_id))

    def process_frames(self):
        # return one or more frames of the sensor
        frames = self._board.get_board_data()

        # for frame in frames.T:
        #     self.push_frame(frame)
        # TODO: fine push chunk error
        # frames = np.transpose(frames)
        # tesst = [[rand() for chan_ix in range(24)]
        #            for samp_ix in range(6)]
        # # frames = frames.astype('float32')
        # if frames.shape[0] > 0:
        #     self.push_frame(samples=frames)

        return frames

    def stop_sensor(self):
        try:
            self._board.stop_stream()
            print('OpenBCIInterface: stopped streaming.')
            self._board.release_session()
            print('OpenBCIInterface: released session.')
        except brainflow.board_shim.BrainFlowError as e:
            print(e)

    # def create_lsl(self, name='OpenBCI_Cyton_8', type='EEG',
    #                nominal_srate=250.0, channel_format='float32',
    #                source_id='Cyton_0'):
    #
    #     channel_count = self.board.get_num_rows(self.board_id)
    #     self.info_eeg = StreamInfo(name=name, type=type, channel_count=channel_count,
    #                                nominal_srate=nominal_srate, channel_format=channel_format,
    #                                source_id=source_id)
    #
    #     # chns = self.info_eeg.desc().append_child('channels')
    #     #
    #     # self.labels = ['Fp1', 'Fp2', 'C3', 'C4', 'T5', 'T6', 'O1', 'O2']
    #     #
    #     # for label in self.labels:
    #     #     ch = chns.append_child("channel")
    #     #     ch.append_child_value('label', label)
    #     #     ch.append_child_value('unit', 'microvolts')
    #     #     ch.append_child_value('type', 'EEG')
    #     #
    #     # self.info_eeg.desc().append_child_value('manufacturer', 'OpenBCI Inc.')
    #     self.outlet_eeg = StreamOutlet(self.info_eeg)
    #
    #     print("--------------------------------------\n" + \
    #           "LSL Configuration: \n" + \
    #           "  Stream 1: \n" + \
    #           "      Name: " + name + " \n" + \
    #           "      Type: " + type + " \n" + \
    #           "      Channel Count: " + str(channel_count) + "\n" + \
    #           "      Sampling Rate: " + str(nominal_srate) + "\n" + \
    #           "      Channel Format: " + channel_format + " \n" + \
    #           "      Source Id: " + source_id + " \n")

    # def push_frame(self, samples):
    #     self.outlet_eeg.push_sample(samples)

    def info_print(self):
        print("Board Information:")
        print("Sampling Rate:", self._board.get_sampling_rate(self.board_id))
        print("Board Id:", self._board.get_board_id())
        print("EEG names:", self._board.get_eeg_names(self.board_id))
        print("Package Num Channel: ", self._board.get_package_num_channel(self.board_id))
        print("EEG Channels:", self._board.get_eeg_channels(self.board_id))
        print("Accel Channels: ", self._board.get_accel_channels(self.board_id))
        print("Other Channels:", self._board.get_other_channels(self.board_id))
        print("Analog Channels: ", self._board.get_analog_channels(self.board_id))
        print("TimeStamp: ", self._board.get_timestamp_channel(self.board_id))
        print("Marker Channel: ", self._board.get_marker_channel(self.board_id))

    def get_sampling_rate(self):
        return self._board.get_sampling_rate(self.board_id)
# def run_test():
#     # data = np.empty(shape=(24, 0))
#     print('Started streaming')
#     start_time = time.time()
#     notch = RealtimeNotch(w0=60, Q=25, fs=250, channel_num=8)
#     butter_bandpass = RealtimeButterBandpass(lowcut=5, highcut=50, fs=250, order=5, channel_num=24)
#     vrms_converter = RealtimeVrms(fs=250, channel_num=8, interval_ms=500, offset_ms=0)
#
#     # starting time
#     start_time = time.time()
#     while 1:
#         try:
#             new_data = openBCI_interface.process_frames()
#             # data = np.concatenate((data, new_data), axis=-1)  # get all data and remove it from internal buffer
#             for data in new_data.T:
#                 eeg_data = data[1:9]
#                 aux_data = data[9:12]
#                 # ######### notch and butter
#                 eeg_data = notch.process_sample(eeg_data)
#                 eeg_data = butter_bandpass.process_sample(eeg_data)
#                 eeg_data = vrms_converter.process_sample(eeg_data)
#                 # push sample to lsl with interval
#                 # if time.time() - start_time > 0.35:
#                 openBCI_interface.push_frame(samples=eeg_data)
#                 # print(eeg_data)
#                 # start_time = time.time()
#         except KeyboardInterrupt:
#             # f_sample = data.shape[-1] / (time.time() - start_time)
#             print('Stopped streaming, sampling rate = ' + str())
#             break
#     return True
#
#
# def run_test_lsl():
#     while 1:
#         try:
#             openBCI_interface.process_frames()
#         except KeyboardInterrupt:
#             # f_sample = data.shape[-1] / (time.time() - start_time)
#             print('Stopped streaming, sampling rate = ' + str())
#             break
#     return True
#
#
# if __name__ == "__main__":
#     openBCI_interface = OpenBCILSLInterface()
#     openBCI_interface.create_lsl()
#     openBCI_interface.start_sensor()
#     data = run_test_lsl()
#     openBCI_interface.stop_sensor()
