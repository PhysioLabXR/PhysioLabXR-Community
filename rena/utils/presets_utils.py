from rena import config
from rena.settings.Presets import Presets, PresetType


def get_preset_category(preset_name):
    preset = Presets()
    if preset_name in preset.experiment_presets.keys():
        return PresetType.EXPERIMENT
    else:
        return preset[preset_name].preset_type


def get_all_preset_names():
    return Presets().keys()


def get_stream_preset_names():
    return Presets().stream_presets.keys()


def get_experiment_preset_streams(exp_name):
    return Presets().experiment_presets[exp_name]

