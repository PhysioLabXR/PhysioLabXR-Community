print("!!! importing os.path")
import os.path

print("!!! importing pyqtgraph")
#import pyqtgraph

print("!!! from PyQt6 import QtCore")
from PyQt6.QtCore import Qt

print("!!! from PyQt6.QtGui import QPixmap")
from PyQt6.QtGui import QPixmap

print("!!! from PyQt6.QtWidgets import QLabel")
from PyQt6.QtWidgets import QLabel

print("!!! from physiolabxr.configs.configs import GlobalSignals")
from physiolabxr.configs.GlobalSignals import GlobalSignals

print("!!! from physiolabxr.configs.configs import AppConfigs")
from physiolabxr.configs.configs import AppConfigs

print("!!! from from physiolabxr.presets.Presets import Presets")
from physiolabxr.presets.Presets import Presets

print("!!! from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier")
from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier

print("!!! from physiolabxr.configs import config_ui, config")
from physiolabxr.configs import config_ui, config

print("!!! from physiolabxr.ui.dialogs import dialog_popup")
from physiolabxr.ui.dialogs import dialog_popup

print("!!!SplashLoadingTextNotifier().set_loading_text('Loading presets...')")
default_settings_dict = {'theme': config_ui.default_theme}
def load_settings(revert_to_default=True, reload_presets=True, reload_configs=True):
    print("!!!SplashLoadingTextNotifier().set_loading_text('Loading presets...')")
    SplashLoadingTextNotifier().set_loading_text('Loading presets...')
    print("Settings are stored at {0}".format(config.settings.fileName()))
    print("!!!SplashLoadingTextNotifier().set_loading_text('Loading settings...')")
    if revert_to_default:
        config.settings.setValue('theme', config_ui.default_theme)
        print("!!!config.settings.setValue('theme', config_ui.default_theme)")
        load_default_recording_file_location()
        print("!!!load_default_recording_file_location()")
    else:
        if not config.settings.contains('theme') or config.settings.value('theme') is None:
            config.settings.setValue('theme', config_ui.default_theme)
            print("!!!config.settings.setValue('theme', config_ui.default_theme)")
        if not config.settings.contains('recording_file_location') or config.settings.value('recording_file_location') is None:
            load_default_recording_file_location()
            print("!!!load_default_recording_file_location()")
    config.settings.sync()
    print("!!!config.settings.sync()")
    # load the presets, reload from local directory the default LSL, device and experiment presets
    preset_root = AppConfigs()._preset_path
    print("!!!preset_root = AppConfigs()._preset_path")

    Presets(_preset_root=preset_root, _reset=reload_presets)  # create the singleton presets object
    print("!!!Presets(_preset_root=preset_root, _reset=reload_presets)")

    # instantiate the GlobalSignals singleton object
    GlobalSignals()
    print("!!!GlobalSignals()")
    #pyqtgraph.setConfigOptions(useNumba=True, useOpenGL=False)
    print("!!!pyqtgraph.setConfigOptions(useNumba=True, useOpenGL=True, enableExperimental=True)")

def load_ui_shared():
    global stream_unavailable_pixmap
    global stream_available_pixmap
    global stream_active_pixmap
    stream_unavailable_pixmap = QPixmap('_media/icons/streamwidget_stream_unavailable.png')
    stream_available_pixmap = QPixmap('_media/icons/streamwidget_stream_available.png')
    stream_active_pixmap = QPixmap('_media/icons/streamwidget_stream_viz_active.png')

def show_splash():
    splash = QLabel()
    pixmap = QPixmap('_media/logo/app_logo.png')
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




