import json
import os


def slice_len_for(slc, seqlen):
    start, stop, step = slc.indices(seqlen)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)

def load_all_LSL_presets(lsl_preset_roots='LSLPresets'):
    preset_file_names = os.listdir(lsl_preset_roots)
    preset_file_paths = [os.path.join(lsl_preset_roots, x) for x in preset_file_names]
    presets = {}
    for pf_path in preset_file_paths:
        preset_dict = json.load(open(pf_path))

        try:
            assert preset_dict['NumChannels'] == len(preset_dict['ChannelNames'])
        except AssertionError:
            raise Exception('Unable to load {0}, number of channels mismatch the number of channel names.'.format(pf_path))

        stream_name = preset_dict.pop('StreamName')
        presets[stream_name] = preset_dict

    return presets

