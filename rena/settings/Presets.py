import json
import os
from collections import defaultdict
from typing import Dict, Any, List, DefaultDict
from dataclasses import dataclass, field, asdict

from PyQt5.QtCore import QStandardPaths

from rena import config
from rena.config import app_data_name
from rena.settings.GroupEntry import GroupEntry
from rena.settings.PlotConfig import PlotConfig


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
        if isinstance(obj, _StreamPreset):
            return obj.to_dict()
        if isinstance(obj, PlotConfig):
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

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Presets:
    """
    dataclass containing all the presets

    Attributes:
        stream_presets: dictionary containing all the stream presets. The key is the stream name and the value is the StreamPreset object
        experiment_presets: dictionary containing all the experiment presets. The key is the experiment name and the value is a list of stream names
    """
    stream_presets: Dict[str, _StreamPreset] = field(default_factory=dict)
    experiment_presets: Dict[str, list] = field(default_factory=dict)
    app_data_path: str = os.path.join(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation), app_data_name)

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

    def sync_local(self):
        if not os.path.exists(self.app_data_path):
            os.makedirs(self.app_data_path)
        path = os.path.join(self.app_data_path, 'Presets.json')
        json_data = json.dumps(self.__dict__, indent=4, cls=PresetsEncoder)
        with open(path, 'w') as f:
            f.write(json_data)