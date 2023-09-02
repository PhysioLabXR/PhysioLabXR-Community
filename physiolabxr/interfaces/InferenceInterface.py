import numpy as np

from pylsl import StreamInfo, StreamOutlet, local_clock, StreamInlet, resolve_byprop
from physiolabxr.configs import config
from physiolabxr.utils.sim import sim_inference


class InferenceInterface:

    def __init__(self, lsl_data_name=config.INFERENCE_LSL_NAME, lsl_data_type=config.INFERENCE_LSL_TYPE):  # default board_id 2 for Cyton
        self.lsl_data_type = lsl_data_type
        self.lsl_data_name = lsl_data_name

        # TODO need to change the channel count when adding eeg
        info = StreamInfo(lsl_data_name, lsl_data_type, channel_count=config.EYE_TOTAL_POINTS_PER_INFERENCE, channel_format='float32', source_id='myuid2424')
        info.desc().append_child_value("apocalyvec", "RealityNavigation")

        # chns = info.desc().append_child("eeg_channels")
        # channel_names = ["C3", "C4", "Cz", "FPz", "POz", "CPz", "O1", "O2", '1','2','3','4','5','6','7','8']
        # for label in channel_names:
        #     ch = chns.append_child("channel")
        #     ch.append_child_value("label", label)
        #     ch.append_child_value("unit", "microvolts")
        #     ch.append_child_value("type", "EEG")

        chns = info.desc().append_child("eye")
        channel_names = ['left_pupil_diameter_sample', 'right_pupil_diameter_sample']
        for label in channel_names:
            ch = chns.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "mm")
            ch.append_child_value("type", "eye")

        self.outlet = StreamOutlet(info, max_buffered=360)
        self.start_time = local_clock()

        self.inlet = None
        self.connect_inference_result_stream()


    def connect_inference_result_stream(self):
        streams = resolve_byprop('type', config.INFERENCE_LSL_RESULTS_TYPE, timeout=1)

        if len(streams) == 0:
            print('No scripting stream open.')
        else:  # TODO handle external scripting stream lost
            self.inlet = StreamInlet(streams[0])
            self.inlet.open_stream()

    def disconnect_inference_result_stream(self):
        self.inlet.close_stream()

    def send_samples_receive_inference(self, samples_dict):
        """
        receive frames
        :param frames:
        """
        # TODO add EEG
        sample = np.reshape(samples_dict['eye'], newshape=(-1, ))  # flatten out
        sample = sample.tolist()  # have to convert to list for LSL

        # chunk[0][0] = 42.0
        # chunk[0][1] = 24.0

        self.outlet.push_sample(sample)

        if self.inlet:
            inference_results_moving_averaged, timestamps = self.inlet.pull_chunk()
            return inference_results_moving_averaged
        else:
            return sim_inference()
