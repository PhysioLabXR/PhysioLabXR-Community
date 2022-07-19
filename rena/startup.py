import os.path

from rena import config, config_ui
from rena.utils.general import *
from rena.utils.settings_utils import export_preset_to_settings
from rena.utils.ui_utils import dialog_popup


def load_default_settings():
    print("Settings are stored at {0}".format(config.settings.fileName()))
    if not config.settings.contains('theme') or config.settings.value('theme') is None:
        config.settings.setValue('theme', config_ui.default_theme)
    if not config.settings.contains('file_format') or config.settings.value('file_format') is None:
        config.settings.setValue('file_format', config.DEFAULT_FILE_FORMAT)
    if not config.settings.contains('recording_file_location') or config.settings.value('recording_file_location') is None:
        config.settings.setValue('recording_file_location', config.DEFAULT_DATA_DIR)
        if not os.path.isdir(config.settings.value('recording_file_location')):
            try:
                os.mkdir(config.settings.value('recording_file_location'))
            except FileNotFoundError:
                dialog_popup(msg='Unable to create recording file location at {0}. '
                                 'Please go to File->Settings and set the the recording file save location before'
                                 'start recording.'.format(config.settings.value('recording_file_location')), title='Warning')
        print("Using default recording location {0}".format(config.settings.value('recording_file_location')))

    print('Reloading presets from Preset directory to persistent settings')
    # load the presets, reload from local directory the default LSL, device and experiment presets
    config.settings.remove('presets')  # TODO: in production, change this to change if preset changed on file system
    LSLStream_presets_dict = load_all_lslStream_presets()
    device_presets_dict = load_all_Device_presets()
    experiment_presets_dict = load_all_experiment_presets()
    # add plot groups
    LSLStream_presets_dict = dict([(stream_name, process_plot_group(preset)) for stream_name, preset in LSLStream_presets_dict.items()])
    device_presets_dict = dict([(stream_name, process_plot_group(preset)) for stream_name, preset in device_presets_dict.items()])

    [export_preset_to_settings(p, 'experimentpresets') for p in experiment_presets_dict.items()]
    [export_preset_to_settings(p, 'streampresets') for p in {**LSLStream_presets_dict, **device_presets_dict}.values()]

    config.settings.sync()

    print('Loading avaiable cameras')
    cameras = get_working_camera_ports()
    cameras = list(map(str, cameras[1]))
    config.settings.setValue('cameras', cameras + ['monitor1'])
