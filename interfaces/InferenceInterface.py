import numpy as np

from pylsl import StreamInfo, StreamOutlet, local_clock
import config

class InferenceInterface:

    def __init__(self, lsl_data_name=config.INFERENCE_LSL_NAME, lsl_data_type=config.INFERENCE_LSL_TYPE):  # default board_id 2 for Cyton
        self.lsl_data_type = lsl_data_type
        self.lsl_data_name = lsl_data_name

        # TODO need to change the channel count when adding eeg
        info = StreamInfo(lsl_data_name, lsl_data_type, channel_count=2, channel_format='float32', source_id='myuid2424')
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

    def send_samples_receive_inference(self, samples_dict):
        """
        receive frames
        :param frames:
        """
        # TODO add EEG
        chunk = np.reshape(samples_dict['eye'], newshape=(-1, samples_dict['eye'].shape[-1]))
        chunk = chunk.tolist()  # have to convert to list for LSL
        self.outlet.push_chunk(chunk)

        return 1, 2