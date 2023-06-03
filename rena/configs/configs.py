import json
import os
from dataclasses import dataclass
from enum import Enum

from PyQt5.QtCore import QStandardPaths

from rena.utils.ConfigPresetUtils import reload_enums, save_local
from rena.utils.Singleton import Singleton


class LinechartVizMode(Enum):
    INPLACE = "in place"
    CONTINUOUS = "continuous"


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
    app_data_path = os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), _app_data_name)

    linechart_viz_mode: LinechartVizMode = LinechartVizMode.INPLACE

    def __post_init__(self):
        self._app_config_path: str = os.path.join(self.app_data_path, self._file_name)

        if not self._reset and os.path.exists(self._app_config_path):
            self.__dict__.update({(k, v) for k, v in json.load(open(self._app_config_path, 'r')).items() if not k.startswith('_')})
        reload_enums(self)

    def __del__(self):
        """
        save the presets to the local disk when the application is closed
        """
        save_local(self.app_data_path, self.__dict__, self._file_name, encoder=AppConfigsEncoder)
        print(f"AppConfigs instance successfully deleted with its contents saved to {self._app_config_path}")

    def get_app_data_path(self):
        return os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), self._app_data_name)
