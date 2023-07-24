import json
import os
import sys
import warnings
from dataclasses import dataclass, fields
from enum import Enum

from PyQt6.QtCore import QStandardPaths
from PyQt6.QtGui import QIcon

from rena.utils.ConfigPresetUtils import reload_enums, save_local
from rena.utils.Singleton import Singleton


class LinechartVizMode(Enum):
    INPLACE = "in place"
    CONTINUOUS = "continuous"


class RecordingFileFormat(Enum):
    dats = "data arrays and timestamps (.dats)"
    pickle = "pickle (.p)"
    matlab = "matlab (.m)"
    csv = "comma separated values (.csv)"
    xdf = "extended data format (.xdf)"

    def get_file_extension(self):
        return self.value.split('(')[1].strip(')')

    @classmethod
    def get_default_file_extension(cls):
        return cls.dats.get_file_extension()


class AppConfigsEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle enums and objects whose metaclass is SubPreset.

    Note all subclass under presets should have their metaclass be SubPreset. So that the encoder can handle them when
    json serialization is called.

    NOTE: if a SubPreset class has a field that is an enum,
    """

    def default(self, o):
        if isinstance(o, Enum):
            return o.name
        if isinstance(o,QIcon):
            return None
        return super().default(o)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class AppConfigs(metaclass=Singleton):
    """
    Global configuration for RenaLabApp. This is a singleton class. It will be created when the application is started.
    Note that AppConfigs must be created before Presets. So in main.py, any imports that involves Presets must be placed
    after AppConfigs is created.

    To add a new config, simply add a new field to this class. Type can be any atomic type and enums. However, the current
    version does not support adding a custom class as a field.

    For example, as of 6/2/2023, the following import order is correct:
    '''
        from rena.config import app_logo_path
        from rena.configs.configs import AppConfigs

        AppConfigs(_reset=False)  # create the singleton app configs object
        from MainWindow import MainWindow
        from rena.startup import load_settings
    '''
    """
    _app_config_path: str = None
    _reset: bool = False
    _file_name = 'AppConfigs.json'
    _app_data_name: str = 'RenaLabApp'
    app_data_path = os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation), _app_data_name)

    linechart_viz_mode: LinechartVizMode = LinechartVizMode.INPLACE
    recording_file_format: RecordingFileFormat = RecordingFileFormat.dats
    eviction_interval: int = 1000

    max_timeseries_num_channels_per_group = int(2 ** 10)
    viz_buffer_max_size = int(2 ** 18)

    visualization_refresh_interval: int = 20  # in milliseconds, how often does the visualization refreshes
    pull_data_interval: int = 2  # in milliseconds, how often does the sensor/LSL pulls data from their designated sources
    video_device_refresh_interval: int = 33

    # path_dict = {'splash_screen_path': 'media/logo/splash_screen.png',
    #              'stream_unavailable': 'media/logo/streamwidget_stream_unavailable.png',
    #              'stream_available': 'media/logo/streamwidget_stream_unavailable.png',
    #              'stream_viz_active': 'media/logo/streamwidget_stream_viz_active.png',
    #              }

    _icons_path = 'media/icons'
    _logo_path = 'media/logo'

    _ui_path = 'ui'
    _ui_file_tree_depth = 3
    _preset_path = 'Presets'
    _rena_base_script = "rena/scripting/BaseRenaScript.py"

    def __post_init__(self):
        # change the cwd to root folder
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        print(f"AppConfigs: current working directory is changed to {os.getcwd()}")
        self._app_config_path: str = os.path.join(self.app_data_path, self._file_name)

        if not self._reset and os.path.exists(self._app_config_path):
            try:
                self.__dict__.update({(k, v) for k, v in json.load(open(self._app_config_path, 'r')).items() if not k.startswith('_')})
            except json.decoder.JSONDecodeError:
                warnings.warn("AppConfigs: failed to load configs from file. Resetting AppConfigs to default values.")

        icon_file_paths = {x: os.path.join(self._icons_path, x) for x in os.listdir(self._icons_path)}
        logo_file_paths = {x: os.path.join(self._logo_path, x) for x in os.listdir(self._logo_path)}
        for file_name, icon_logo_path in {**icon_file_paths, **logo_file_paths}.items():
            assert not hasattr(self, file_name), f"found duplicate file name {file_name} in AppConfigs's attributes"
            file_name, ext = os.path.splitext(file_name)
            name = f'_{file_name}'
            setattr(self, name, icon_logo_path)
            if ext == '.svg':
                setattr(self, f'_icon_{file_name}', QIcon(icon_logo_path))

        ui_file_paths = self._load_ui_files_recursive(self._ui_path, self._ui_file_tree_depth)
        for file_name, ui_path in ui_file_paths.items():
            name = f'_ui_{file_name}'
            assert not hasattr(self, name), f"found duplicate file name {file_name} in AppConfigs's attributes"
            setattr(self, name, ui_path)
        self._rena_base_script = open(self._rena_base_script, "r").read()
        reload_enums(self)

    def __del__(self):
        """
        save the presets to the local disk when the application is closed
        """
        save_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        save_local(self.app_data_path, save_dict, self._file_name, encoder=AppConfigsEncoder)
        print(f"AppConfigs instance successfully deleted with its contents saved to {self._app_config_path}")

    def get_app_data_path(self):
        return os.path.join(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation), self._app_data_name)

    def revert_to_default(self):
        for field in fields(self):
            if not field.name.startswith("_"):
                setattr(self, field.name, field.default)

    def _load_ui_files_recursive(self, path, depth):
        ui_file_paths = {}
        for x in os.listdir(path):
            full_path = os.path.join(path, x)
            if os.path.isdir(full_path) and depth > 1:
                # Recursively traverse the subdirectory with reduced depth
                ui_file_paths.update(self._load_ui_files_recursive(full_path, depth - 1))
            elif x.endswith('.ui'):
                ui_file_paths[os.path.splitext(x)[0]] = full_path

        return ui_file_paths