import json
import multiprocessing
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List

from PyQt5.QtCore import QStandardPaths

from rena import config
from rena.config import app_data_name
from rena.settings.GroupEntry import GroupEntry
from rena.settings.PlotConfig import PlotConfig
from rena.utils.Singleton import Singleton
from rena.utils.fs_utils import get_file_changes_multiple_dir
from rena.utils.settings_utils import validate_preset_json_preset, process_plot_group_json_preset


# class DataclassEncoder(json.JSONEncoder):
#     def default(self, o: Any) -> Any:
#         if hasattr(o, '__dict__'):
#             return o.__dict__
#         elif hasattr(o, '__dataclass_fields__'):
#             return asdict(o)
#         else:
#             return super().default(o)
class PresetsEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, _StreamPreset) or isinstance(obj, PlotConfig):
            return obj.to_dict()
        return super().default(obj)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class _StreamPreset:
    """
    Stream preset defines a stream to be loaded from the GUI.

    IMPORTANT: the only entry point to create stream preset is through the add_stream_preset function in the Presets class.
    attributes:
        stream_name: name of the stream
        channel_names: list of channel names
        num_channels: number of channels in the stream

        group_info: dictionary containing the group information. The key is the group name and the value is the GroupEntry object.
        This attribute directs how the stream should be displayed in the GUI.

        device_info: dictionary containing the device information. The key is the device name and the value is the device information.
        This attribute is used to connect to the devices that are not hooked to LSL or ZMQ. It contains information necessary for
        the connection. While LSL finds stream by StreamName, and ZMQ finds stream by IP address and port number, the device_info
        contains information that is specific to the connection of a device such as the serial port, baud rate, etc.

        networking_interface: name of the networking interface to use to receive the stream. This is set from the Presets
        class when calling the add_stream_preset function. If from json preset file loaded at startup, this information
        is obtained by which folder the preset file is in (i.e., LSLPresets, ZMQPresets, or DevicePresets under the Presets folder)
    """
    stream_name: str
    channel_names: List[str]

    num_channels: int

    group_info: dict[str, GroupEntry]
    device_info: dict
    networking_interface: str

    data_type: str = 'float32'

    port_number: int = None

    display_duration: float = None
    nominal_sampling_rate: int = 1

    def __post_init__(self):
        """
        StreamPreset's post init function. It will set the display_duration attribute based on the default_display_duration in the config file.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        @return:
        """
        if self.display_duration is None:
            self.display_duration = config.settings.value('default_display_duration')

    def to_dict(self) -> dict[str, Any]:
        """
        auto loop the attributes converting them to dictionary, in case of the GroupEntry object, it will call the asdict function
        because GroupEntry is a dataclass nested under the StreamPreset class.
        Devs: if you add another attribute that's a nested dataclass, you will need to add the todict call here, like how it's done for the GroupEntry
        @return:  dictionary containing all the attributes of the StreamPreset class
        """
        return {attr: {group_name: group_entry.__dict__ for group_name, group_entry in value.items()} if attr == 'group_info' else value
                for attr, value in self.__dict__.items()}

def save_local(app_data_path, preset_dict) -> None:
    """
    sync the presets to the local disk. This will create a Presets.json file in the app data folder if it doesn't exist.
    applies file lock while the json is being dumped. This will block another other process from accessing the file without
    raising an exception.
    """
    if not os.path.exists(app_data_path):
        os.makedirs(app_data_path)
    path = os.path.join(app_data_path, 'Presets.json')
    json_data = json.dumps(preset_dict, indent=4, cls=PresetsEncoder)

    with open(path, 'w') as f:
        f.write(json_data)


def _load_stream_presets(presets, dirty_presets):
    for category, dirty_preset_paths in dirty_presets.items():
        for dirty_preset_path in dirty_preset_paths:
            loaded_preset_dict = json.load(open(dirty_preset_path))

            if category == 'LSL' or category == 'ZMQ' or category == 'Device':
                stream_preset_dict = validate_preset_json_preset(loaded_preset_dict)
                stream_preset_dict['networking_interface'] = category
                stream_preset_dict = process_plot_group_json_preset(stream_preset_dict)
                presets.add_stream_preset(stream_preset_dict)
            elif category == 'Experiment':
                presets.add_experiment_preset(loaded_preset_dict['ExperimentName'], loaded_preset_dict['PresetStreamNames'])
            else:
                raise ValueError(f'unknown category {category} for preset {dirty_preset_path}')


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Presets(metaclass=Singleton):
    """
    dataclass containing all the presets

    Attributes:
        stream_presets: dictionary containing all the stream presets. The key is the stream name and the value is the StreamPreset object
        experiment_presets: dictionary containing all the experiment presets. The key is the experiment name and the value is a list of stream names

        _reset: if true, reload the presets. This is done in post_init by removing files at _last_mod_time_path and _preset_path.
    """
    _preset_root: str = None
    _reset: bool = False

    _lsl_preset_root: str = 'LSLPresets'
    _zmq_preset_root: str = 'ZMQPresets'
    _device_preset_root: str = 'DevicePresets'
    _experiment_preset_root: str = 'ExperimentPresets'

    stream_presets: Dict[str, _StreamPreset] = field(default_factory=dict)
    experiment_presets: Dict[str, list] = field(default_factory=dict)
    _app_data_path: str = os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), app_data_name)
    _last_mod_time_path: str = os.path.join(_app_data_path, 'last_mod_times.json')
    _preset_path: str = os.path.join(_app_data_path, 'Presets.json')

    def __post_init__(self):
        """
        1. load the presets from the local disk if it exists
        """
        if self._preset_root is None:
            raise ValueError('preset root must not be None when first time initializing Presets')
        if self._reset:
            if os.path.exists(self._last_mod_time_path):
                os.remove(self._last_mod_time_path)
            if os.path.exists(self._preset_path):
                os.remove(self._preset_path)

        self._lsl_preset_root = os.path.join(self._preset_root, self._lsl_preset_root)
        self._zmq_preset_root = os.path.join(self._preset_root, self._zmq_preset_root)
        self._device_preset_root = os.path.join(self._preset_root, self._device_preset_root)
        self._experiment_preset_root = os.path.join(self._preset_root, self._experiment_preset_root)
        self._preset_roots = [self._lsl_preset_root, self._zmq_preset_root, self._device_preset_root, self._experiment_preset_root]

        if os.path.exists(self._preset_path):
            print(f'Reloading presets from {self._app_data_path}')
            with open(self._preset_path, 'r') as f:
                preset_dict = json.load(f)
                self.__dict__.update(preset_dict)
        dirty_presets = self._record_presets_last_modified_times()

        _load_stream_presets(self, dirty_presets)

        self.save_async()
        print("Presets instance successfully initialized")

    def _record_presets_last_modified_times(self):
        """
        get all the dirty presets and record their last modified times to the last_mod_times.json file.
        This will be called when the application is opened, because this is when the presets are loaded from the local disk.
        @return:
        """
        if os.path.exists(self._last_mod_time_path):
            with open(self._last_mod_time_path, 'r') as f:
                last_mod_times = json.load(f)
        else:  # if the last_mod_times.json doesn't exist, then all the presets are dirty
            last_mod_times = {}  # passing empty last_mod_times to the get_file_changes_multiple_dir function will return all the files

        dirty_presets = {'LSL': None, 'ZMQ': None, 'Device': None, 'Experiment': None}
        (dirty_presets['LSL'], dirty_presets['ZMQ'], dirty_presets['Device'], dirty_presets['Experiment']), current_mod_times = get_file_changes_multiple_dir(self._preset_roots, last_mod_times)

        with open(self._last_mod_time_path, 'w') as f:
            json.dump(current_mod_times, f)

        return dirty_presets

    def __del__(self):
        """
        save the presets to the local disk when the application is closed
        """
        save_local(self._app_data_path, self.__dict__)

    def add_stream_preset(self, stream_preset_dict: Dict[str, Any]):
        """
        add a stream preset to the presets

        checks for any attributes in the stream_preset_dict that are not in the StreamPreset class. These are assumed
        to be device specific attributes and it is recommended to give some prefix for these attributes (e.g. '_', '_device_')
        :param stream_preset_dict: dictionary containing the stream preset
        :return: None
        """
        device_info = {}
        device_specific_attribute_names = [attribute_name for attribute_name, attribute_value in stream_preset_dict.items() if attribute_name not in _StreamPreset.__annotations__]
        for attribute_name in device_specific_attribute_names:
            device_info[attribute_name] = stream_preset_dict.pop(attribute_name)
        stream_preset_dict['device_info'] = device_info

        stream_preset = _StreamPreset(**stream_preset_dict)
        self.stream_presets[stream_preset.stream_name] = stream_preset


    def add_experiment_preset(self, experiment_name: str, stream_names: List[str]):
        """
        add an experiment preset to the presets
        :param experiment_name: name of the experiment
        :param stream_names: list of stream names
        :return: None
        """
        self.experiment_presets[experiment_name] = stream_names

    def save_async(self) -> None:
        """
        save the presets to the local disk asynchronously.
        @return: None
        """
        p = multiprocessing.Process(target=save_local, args=(self._app_data_path, self.__dict__))
        p.start()