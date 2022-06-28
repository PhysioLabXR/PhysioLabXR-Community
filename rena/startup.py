import os.path

from rena import config, config_ui
from rena.utils.ui_utils import dialog_popup


def load_default_settings():
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
