import json
import os

import numpy as np


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)


def load_all_LSL_presets(lsl_preset_roots='LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        loaded_preset_dict = json.load(open(pf_path))

        if 'ChannelNames' in loaded_preset_dict.keys():
            try:
                assert loaded_preset_dict['NumChannels'] == len(loaded_preset_dict['ChannelNames'])
            except AssertionError:
                raise Exception('Unable to load {0}, number of channels mismatch the number of channel names.'.format(pf_path))
        else:
            loaded_preset_dict['ChannelNames'] = ['Unknown'] * loaded_preset_dict['NumChannels']

        stream_name = loaded_preset_dict.pop('StreamName')

        if 'GroupChannelsInPlot' in loaded_preset_dict.keys():
            try:
                assert np.max(loaded_preset_dict['GroupChannelsInPlot']) <= loaded_preset_dict['NumChannels']
            except AssertionError:
                raise Exception('GroupChannelsInPlot max must be less than the number of channels.')

            loaded_preset_dict["PlotGroupSlices"] = []
            head = 0
            for x in loaded_preset_dict['GroupChannelsInPlot']:
                loaded_preset_dict["PlotGroupSlices"].append((head, x))
                head = x
            if head != loaded_preset_dict['NumChannels']:
                loaded_preset_dict["PlotGroupSlices"].append((head, loaded_preset_dict['NumChannels']))  # append the last group
        else:
            loaded_preset_dict["PlotGroupSlices"] = None
        presets[stream_name] = loaded_preset_dict

    return presets

