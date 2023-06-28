import json
import multiprocessing
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Union

import numpy as np
from PyQt6.QtCore import QStandardPaths

from rena import config
from rena.config import app_data_name, default_group_name
from rena.configs.configs import AppConfigs
from rena.presets.GroupEntry import GroupEntry
from rena.presets.preset_class_helpers import SubPreset
from rena.utils.ConfigPresetUtils import save_local, reload_enums
from rena.utils.Singleton import Singleton
from rena.utils.fs_utils import get_file_changes_multiple_dir
from rena.presets.load_user_preset import process_plot_group_json_preset, validate_preset_json_preset
from rena.utils.video_capture_utils import get_working_camera_ports


class DataType(Enum):
    """
    Data types supported by RenaLabApp.
    Calling the enum with a value will return the same value of the  corresponding numpy data type.
    Use the class method get_data_type to get actual data type.
    """
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"

    def __call__(self, *args, **kwargs):
        return self.get_data_type()(args[0])

    def get_data_type(self):
        if self == DataType.uint8:
            return np.uint8
        elif self == DataType.uint16:
            return np.uint16
        elif self == DataType.uint32:
            return np.uint32
        elif self == DataType.uint64:
            return np.uint64
        elif self == DataType.int8:
            return np.int8
        elif self == DataType.int16:
            return np.int16
        elif self == DataType.int32:
            return np.int32
        elif self == DataType.int64:
            return np.int64
        elif self == DataType.float16:
            return np.float16
        elif self == DataType.float32:
            return np.float32
        elif self == DataType.float64:
            return np.float64


class PresetType(Enum):
    WEBCAM = 'WEBCAM'
    MONITOR = 'MONITOR'
    LSL = 'LSL'
    ZMQ = 'ZMQ'
    CUSTOM = 'CUSTOM'
    EXPERIMENT = 'EXPERIMENT'


class VideoDeviceChannelOrder(Enum):
    RGB = 0
    BGR = 1

# class VideoDeviceTypeEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, Enum):
#             return obj.value
#         return json.JSONEncoder.default(self, obj)


# class PresetsEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, StreamPreset) or isinstance(obj, PlotConfigs) or isinstance(obj, VideoPreset):
#             return obj.to_dict()
#         return super().default(obj)

class PresetsEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle enums and objects whose metaclass is SubPreset.

    Note all subclass under presets should have their metaclass be SubPreset. So that the encoder can handle them when
    json serialization is called.

    NOTE: if a SubPreset class has a field that is an enum,
    """

    def default(self, o):
        if isinstance(o, Enum):
            return o.name
        if o.__class__.__class__ is SubPreset:
            return o.__dict__
        return super().default(o)




@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StreamPreset(metaclass=SubPreset):
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
    preset_type: PresetType

    data_type: DataType = DataType.float32
    port_number: int = None
    display_duration: float = None
    nominal_sampling_rate: int = 10

    def __post_init__(self):
        """
        StreamPreset's post init function. It will set the display_duration attribute based on the default_display_duration in the config file.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        @return:
        """
        if self.display_duration is None:
            self.display_duration = float(config.settings.value('viz_display_duration'))
        for key, value in self.group_info.items():  # recreate the GroupEntry object from the dictionary
            if isinstance(value, dict):
                self.group_info[key] = GroupEntry(**value)
        # convert any enum attribute loaded as string to the corresponding enum value
        reload_enums(self)

    def add_group_entry(self, group_entry: GroupEntry):
        assert group_entry.group_name not in self.group_info.keys(), f'Group {group_entry.group_name} already exists in the stream preset'
        self.group_info[group_entry.group_name] = group_entry

    def get_next_available_groupname(self):
        i = 0
        rtn = f'{default_group_name}{i}'
        while rtn in self.group_info.keys():
            i += 1
            rtn = f'{default_group_name}{i}'

        return rtn

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class VideoPreset(metaclass=SubPreset):
    """
    Stream preset defines a stream to be loaded from the GUI.

    IMPORTANT: the only entry point to create stream preset is through the add_stream_preset function in the Presets class.
    attributes:
        stream_name: name of the stream
        video_type: can be webcam or monitor
    """
    stream_name: str
    preset_type: PresetType
    video_id: int

    video_scale: float = 1.0
    channel_order: VideoDeviceChannelOrder = VideoDeviceChannelOrder.RGB

    def __post_init__(self):
        """
        VideoPreset's post init function.
        @return:
        """
        # convert any enum attribute loaded as string to the corresponding enum value
        reload_enums(self)


def _load_stream_presets(presets, dirty_presets):
    for category, dirty_preset_paths in dirty_presets.items():
        for dirty_preset_path in dirty_preset_paths:
            loaded_preset_dict = json.load(open(dirty_preset_path))

            if category == PresetType.LSL.value or category == PresetType.ZMQ.value or category == PresetType.CUSTOM.value:
                stream_preset_dict = preprocess_stream_preset(loaded_preset_dict, category)
                presets.add_stream_preset(stream_preset_dict)
            elif category == PresetType.EXPERIMENT.value:
                presets.add_experiment_preset(loaded_preset_dict['ExperimentName'], loaded_preset_dict['PresetStreamNames'])
            else:
                raise ValueError(f'unknown category {category} for preset {dirty_preset_path}')

def preprocess_stream_preset(stream_preset_dict, category):
    """

    """
    stream_preset_dict = validate_preset_json_preset(stream_preset_dict)
    if type(category) == str:
        preset_type = PresetType[category.upper()]
    elif type(category) == PresetType:
        preset_type = category
    else:
        raise ValueError(f'unknown category {category} for preset {stream_preset_dict} with type {type(category)}')
    stream_preset_dict['preset_type'] = preset_type
    stream_preset_dict = process_plot_group_json_preset(stream_preset_dict)
    return stream_preset_dict


def _load_video_device_presets(presets):
    print('Loading available cameras')
    _, working_camera_ports, _ = get_working_camera_ports()
    working_cameras_stream_names = [f'Camera {x}' for x in working_camera_ports]

    for camera_id, camera_stream_name in zip(working_camera_ports, working_cameras_stream_names):
        presets.add_video_preset(camera_stream_name, PresetType.WEBCAM, camera_id)
    presets.add_video_preset('monitor 0', PresetType.MONITOR, 0)

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

    stream_presets: Dict[str, Union[StreamPreset, VideoPreset]] = field(default_factory=dict)
    experiment_presets: Dict[str, list] = field(default_factory=dict)

    _app_data_path: str = AppConfigs().app_data_path
    _last_mod_time_path: str = os.path.join(_app_data_path, 'last_mod_times.json')
    _preset_path: str = os.path.join(_app_data_path, 'Presets.json')

    def __post_init__(self):
        """
        The post init of presets does the following:
        1. if reset is true, remove the last mod time and preset json files
        2. set the private path variables based on the given preset root, which makes the preset root a mandatory argument when first time initializing Presets globally.
        3. load the presets from the local disk if it exists
        4. check if any presets are dirty and load them
        5. save the presets to the local disk
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
                for key, value in preset_dict['stream_presets'].items():
                    if value['preset_type'] == 'LSL' or value['preset_type'] == 'ZMQ' or value['preset_type'] == 'Device':
                        preset = StreamPreset(**value)
                    elif value['preset_type'] == 'WEBCAM' or value['preset_type'] == 'MONITOR':
                        preset = VideoPreset(**value)
                    preset_dict['stream_presets'][key] = preset

                # for key, value in preset_dict['video_presets'].items():
                #     preset = VideoPreset(**value)
                #     preset_dict['video_presets'][key] = preset
                self.__dict__.update(preset_dict)
        dirty_presets = self._record_presets_last_modified_times()

        _load_stream_presets(self, dirty_presets)
        _load_video_device_presets(self)

        self.save(is_async=True)
        print("Presets instance successfully initialized")

    def _get_all_presets(self):
        """
        this function needs to be modified if new preset dict are added
        """
        return {**self.stream_presets, **self.experiment_presets}

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

        dirty_presets = {PresetType.LSL.value: None, PresetType.ZMQ.value: None, PresetType.CUSTOM.value: None, PresetType.EXPERIMENT.value: None}
        (dirty_presets[PresetType.LSL.value], dirty_presets[PresetType.ZMQ.value], dirty_presets[PresetType.CUSTOM.value], dirty_presets[PresetType.EXPERIMENT.value]), current_mod_times = get_file_changes_multiple_dir(self._preset_roots, last_mod_times)
        if not os.path.exists(self._app_data_path):
            os.makedirs(self._app_data_path)
        with open(self._last_mod_time_path, 'w') as f:
            json.dump(current_mod_times, f)

        return dirty_presets

    def __del__(self):
        """
        save the presets to the local disk when the application is closed
        """
        save_local(self._app_data_path, self.__dict__, 'Presets.json', encoder=PresetsEncoder)
        print(f"Presets instance successfully deleted with its contents saved to {self._app_data_path}")

    def add_stream_preset(self, stream_preset_dict: Dict[str, Any]):
        """
        add a stream preset to the presets

        checks for any attributes in the stream_preset_dict that are not in the StreamPreset class. These are assumed
        to be device specific attributes and it is recommended to give some prefix for these attributes (e.g. '_', '_device_')
        :param stream_preset_dict: dictionary containing the stream preset
        :return: None
        """
        device_info = {}
        device_specific_attribute_names = [attribute_name for attribute_name, attribute_value in stream_preset_dict.items() if attribute_name not in StreamPreset.__annotations__]
        for attribute_name in device_specific_attribute_names:
            device_info[attribute_name] = stream_preset_dict.pop(attribute_name)
        stream_preset_dict['device_info'] = device_info

        stream_preset = StreamPreset(**stream_preset_dict)
        self.stream_presets[stream_preset.stream_name] = stream_preset

    def add_video_preset(self, stream_name, video_type, video_id):
        video_preset = VideoPreset(stream_name, video_type, video_id)
        self.stream_presets[video_preset.stream_name] = video_preset

    def add_experiment_preset(self, experiment_name: str, stream_names: List[str]):
        """
        add an experiment preset to the presets
        :param experiment_name: name of the experiment
        :param stream_names: list of stream names
        :return: None
        """
        self.experiment_presets[experiment_name] = stream_names

    def save(self, is_async=False) -> None:
        """
        save the presets to the local disk asynchronously.
        @return: None
        """
        if is_async:
            p = multiprocessing.Process(target=save_local, args=(self._app_data_path, self.__dict__, 'Presets.json', PresetsEncoder))
            p.start()
        else:
            save_local(self._app_data_path, self.__dict__, 'Presets.json', encoder=PresetsEncoder)

    def __getitem__(self, key):
        return self._get_all_presets()[key]

    def keys(self):
        return self._get_all_presets().keys()


