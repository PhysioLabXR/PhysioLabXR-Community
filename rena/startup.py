import os.path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

from rena.utils.video_capture_utils import get_working_camera_ports
from rena.ui_shared import *
from rena import config, config_ui
from rena.utils.settings_utils import export_preset_to_settings, load_all_presets, \
    load_all_experiment_presets, process_plot_group
from rena.utils.ui_utils import dialog_popup

default_settings_dict = {'theme': config_ui.default_theme,
                         'file_format': config.DEFAULT_FILE_FORMAT}
def load_settings(revert_to_default=True, reload_presets=True):
    print("Settings are stored at {0}".format(config.settings.fileName()))
    if revert_to_default:
        config.settings.setValue('theme', config_ui.default_theme)
        config.settings.setValue('file_format', config.DEFAULT_FILE_FORMAT)
        load_default_recording_file_location()
        config.settings.setValue('viz_data_buffer_max_size', config.VIZ_DATA_BUFFER_MAX_SIZE)
        config.settings.setValue('viz_display_duration', config.VIZ_DISPLAY_DURATION)
        config.settings.setValue('video_device_refresh_interval', config.VIDEO_DEVICE_REFRESH_INTERVAL)
        config.settings.setValue('main_window_meta_data_refresh_interval', config.MAIN_WINDOW_META_DATA_REFRESH_INTERVAL)
        config.settings.setValue('downsample_method_mean_sr_threshold', config.downsample_method_mean_sr_threshold)
        config.settings.setValue('pull_data_interval', config.pull_data_interval)
        config.settings.setValue('visualization_refresh_interval', config.VISUALIZATION_REFRESH_INTERVAL)
        config.settings.setValue('max_timeseries_num_channels', config.MAX_TIMESERIES_NUM_CHANNELS_PER_GROUP)
    else:
        if not config.settings.contains('theme') or config.settings.value('theme') is None:
            config.settings.setValue('theme', config_ui.default_theme)
        if not config.settings.contains('file_format') or config.settings.value('file_format') is None:
            config.settings.setValue('file_format', config.DEFAULT_FILE_FORMAT)
        if not config.settings.contains('recording_file_location') or config.settings.value('recording_file_location') is None:
            load_default_recording_file_location()
        if not config.settings.contains('viz_data_buffer_max_size') or config.settings.value('viz_data_buffer_max_size') is None:
            config.settings.setValue('viz_data_buffer_max_size', config.VIZ_DATA_BUFFER_MAX_SIZE)
        if not config.settings.contains('viz_display_duration') or config.settings.value('viz_display_duration') is None:
            config.settings.setValue('viz_display_duration', config.VIZ_DISPLAY_DURATION)
        if not config.settings.contains('video_device_refresh_interval') or config.settings.value('video_device_refresh_interval') is None:
            config.settings.setValue('video_device_refresh_interval', config.VIDEO_DEVICE_REFRESH_INTERVAL)
        if not config.settings.contains('max_timeseries_num_channels') or config.settings.value('max_timeseries_num_channels') is None:
            config.settings.setValue('max_timeseries_num_channels', config.MAX_TIMESERIES_NUM_CHANNELS_PER_GROUP)
        if not config.settings.contains('downsample_method_mean_sr_threshold') or config.settings.value('downsample_method_mean_sr_threshold') is None:
            config.settings.setValue('downsample_method_mean_sr_threshold', config.downsample_method_mean_sr_threshold)
        if not config.settings.contains('main_window_meta_data_refresh_interval') or config.settings.value('main_window_meta_data_refresh_interval') is None:
            config.settings.setValue('main_window_meta_data_refresh_interval', config.MAIN_WINDOW_META_DATA_REFRESH_INTERVAL)
        if not config.settings.contains('pull_data_interval') or config.settings.value('pull_data_interval') is None:
            config.settings.setValue('pull_data_interval', config.pull_data_interval)
        if not config.settings.contains('visualization_refresh_interval') or config.settings.value('visualization_refresh_interval') is None:
            config.settings.setValue('visualization_refresh_interval', config.VISUALIZATION_REFRESH_INTERVAL)

    print('Reloading presets from Preset directory to persistent settings')
    # load the presets, reload from local directory the default LSL, device and experiment presets
    if reload_presets: config.settings.remove('presets')  # TODO: in production, change this to change if preset changed on file system
    LSL_presets_dict = load_all_presets('../Presets/LSLPresets')
    ZMQ_presets_dict = load_all_presets('../Presets/ZMQPresets')
    device_presets_dict = load_all_presets('../Presets/DevicePresets')
    experiment_presets_dict = load_all_experiment_presets()

    for _, preset in LSL_presets_dict.items():
        preset["NetworkingInterface"] = "LSL"
    for _, preset in ZMQ_presets_dict.items():
        preset["NetworkingInterface"] = "ZMQ"
    for _, preset in device_presets_dict.items():
        preset["NetworkingInterface"] = "Device"

    stream_presets_dict = {**LSL_presets_dict, **ZMQ_presets_dict, **device_presets_dict}  # merge the lsl and device presets
    # add plot groups
    stream_presets_dict = dict([(stream_name, process_plot_group(preset)) for stream_name, preset in stream_presets_dict.items()])

    [export_preset_to_settings(p, 'experimentpresets') for p in experiment_presets_dict.items()]
    [export_preset_to_settings(p, 'streampresets') for p in stream_presets_dict.values()]

    config.settings.sync()

    print('Loading avaiable cameras')
    cameras = get_working_camera_ports()
    cameras = list(map(str, cameras[1]))
    config.settings.setValue('video_device', cameras + ['monitor1'])

def load_ui_shared():
    global stream_unavailable_pixmap
    global stream_available_pixmap
    global stream_active_pixmap
    stream_unavailable_pixmap = QPixmap('../media/icons/streamwidget_stream_unavailable.png')
    stream_available_pixmap = QPixmap('../media/icons/streamwidget_stream_available.png')
    stream_active_pixmap = QPixmap('../media/icons/streamwidget_stream_viz_active.png')

def show_splash():
    splash = QLabel()
    pixmap = QPixmap('../media/logo/RenaLabApp.png')
    splash.setPixmap(pixmap)
    splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    splash.show()
    pass


def load_default_recording_file_location():
    config.settings.setValue('recording_file_location', config.DEFAULT_DATA_DIR)
    if not os.path.isdir(config.settings.value('recording_file_location')):
        try:
            os.mkdir(config.settings.value('recording_file_location'))
        except FileNotFoundError:
            dialog_popup(msg='Unable to create recording file location at {0}. '
                             'Please go to File->Settings and set the the recording file save location before'
                             'start recording.'.format(config.settings.value('recording_file_location')),
                         title='Warning')
    print("Using default recording location {0}".format(config.settings.value('recording_file_location')))
