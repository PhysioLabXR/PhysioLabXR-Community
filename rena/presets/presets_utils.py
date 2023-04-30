from rena.presets.Presets import Presets, PresetType


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


def get_experiment_preset_names():
    return Presets().experiment_presets.keys()


def get_experiment_preset_streams(exp_name):
    return Presets().experiment_presets[exp_name]


def get_stream_preset_info(stream_name, key):
    return Presets().stream_presets[stream_name].__getattribute__(key)


def get_stream_preset_custom_info(stream_name) -> dict:
    return Presets().stream_presets[stream_name].device_info


def set_stream_preset_info(stream_name, key, value):
    setattr(Presets().stream_presets[stream_name], key, value)


def check_preset_exists(stream_name):
    return stream_name in Presets().stream_presets.keys()


def collect_stream_all_groups_info(stream_name):
    """

    @param stream_name: the name of the stream whose group info is to be collected
    @return: a lambda function that returns the group info of the stream, the downstream function should call the lambda function to get the group info
    it returns a lambda instead of static dict because the group info may change during runtime
    """
    return lambda: Presets().stream_presets[stream_name].group_info
