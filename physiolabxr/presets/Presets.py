import copy
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import Process
from typing import Dict, Any, List, Union

import numpy as np
import pyaudio

from physiolabxr.configs import config
from physiolabxr.configs.config import default_group_name
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.GroupEntry import GroupEntry
from physiolabxr.presets.PresetEnums import PresetType, DataType, VideoDeviceChannelOrder, AudioInputDataType
from physiolabxr.presets.ScriptPresets import ScriptPreset, ScriptParam
from physiolabxr.presets.preset_class_helpers import SubPreset
from physiolabxr.ui.SplashScreen import SplashLoadingTextNotifier
from physiolabxr.utils.ConfigPresetUtils import reload_enums
from physiolabxr.utils.Singleton import Singleton
from physiolabxr.utils.dsp_utils.dsp_modules import DataProcessor
from physiolabxr.utils.fs_utils import get_file_changes_multiple_dir
from physiolabxr.presets.load_user_preset import process_plot_group_json_preset, validate_preset_json_preset, \
    create_default_group_info
from physiolabxr.utils.video_capture_utils import get_working_camera_ports


def is_monotonically_increasing(lst):
    differences = np.diff(np.array(lst))
    return np.all(differences >= 0)

class PresetsEncoder(json.JSONEncoder):
    """
    JSON encoder that can handle enums and objects whose metaclass is SubPreset.

    Note all subclass under presets should have their metaclass be SubPreset. So that the encoder can handle them when
    json serialization is called.

    NOTE: if a SubPreset class has a field that is an enum,
    """

    def default(self, o):
        # if dataclasses.is_dataclass(o):
        #     attri_names = list(vars(o).keys())
        #     for key in attri_names:
        #         if key.startswith('_'):
        #             o.__delattr__(key)
        if isinstance(o, Process):
            return None
        if isinstance(o, Enum):
            return o.name
        if isinstance(o, DataProcessor):
            return o.serialize_data_processor_params()
        if o.__class__.__class__ is SubPreset:
            rtn = copy.copy(o.__dict__)
            if isinstance(o, StreamPreset):
                if not o.can_edit_channel_names:  # will not serialize channel names if it is not editable
                    rtn['channel_names'] = None
            if isinstance(o, GroupEntry):
                if o._is_image_only:
                    if o.channel_indices is not None:
                        assert is_monotonically_increasing(o.channel_indices), "channel indices must be monotonically increasing when _is_image_only is True"
                        rtn['channel_indices_start_end'] = min(o.channel_indices), max(o.channel_indices) + 1
                    rtn['channel_indices'] = None
                    rtn['is_channels_shown'] = None
            return rtn
        return super().default(o)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StreamPreset(metaclass=SubPreset):
    """
    Stream preset defines a stream to be loaded from the GUI.

    IMPORTANT: the only entry point to create stream preset is through the add_stream_preset function in the _presets class.
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

        networking_interface: name of the networking interface to use to receive the stream. This is set from the _presets
        class when calling the add_stream_preset function. If from json preset file loaded at startup, this information
        is obtained by which folder the preset file is in (i.e., LSLPresets, ZMQPresets, or DevicePresets under the _presets folder)
    """
    stream_name: str

    num_channels: int

    group_info: dict[str, GroupEntry]
    device_info: dict
    preset_type: PresetType

    channel_names: List[str] = None
    data_type: DataType = DataType.float32
    port_number: int = None
    display_duration: float = None
    nominal_sampling_rate: int = 10

    can_edit_channel_names: bool = True

    data_processor_only_apply_to_visualization: bool = False

    def __post_init__(self):
        """
        StreamPreset's post init function. It will set the display_duration attribute based on the default_display_duration in the config file.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        @return:
        """
        if self.channel_names is None:  # channel names can be none when the number of channels is too big
            self.channel_names = ['c {0}'.format(i) for i in range(self.num_channels)]
            self.can_edit_channel_names = False
            print(f"StreamPreset: disabling channel editing for stream {self.stream_name}")
        if self.num_channels > config.MAX_TS_CHANNEL_NUM:
            self.can_edit_channel_names = False
            print(f"StreamPreset: disabling channel editing for stream {self.stream_name}")
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

    IMPORTANT: the only entry point to create stream preset is through the add_stream_preset function in the _presets class.
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

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class AudioPreset(metaclass=SubPreset):
    """
    Stream preset defines a stream to be loaded from the GUI.

    IMPORTANT: the only entry point to create stream preset is through the add_stream_preset function in the _presets class.
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

        networking_interface: name of the networking interface to use to receive the stream. This is set from the _presets
        class when calling the add_stream_preset function. If from json preset file loaded at startup, this information
        is obtained by which folder the preset file is in (i.e., LSLPresets, ZMQPresets, or DevicePresets under the _presets folder)
    """
    stream_name: str

    ################ audio device fields ################
    audio_device_index:int
    ################ audio device fields ################

    num_channels: int

    group_info: dict[str, GroupEntry]
    device_info: dict

    audio_device_data_format: AudioInputDataType = AudioInputDataType.paInt16
    audio_device_frames_per_buffer: int = 128
    audio_device_sampling_rate: float = 8192

    preset_type: PresetType = PresetType.AUDIO

    channel_names: List[str] = None
    data_type: DataType = DataType.float32
    port_number: int = None
    display_duration: float = None
    nominal_sampling_rate: int = 8192

    can_edit_channel_names: bool = True

    data_processor_only_apply_to_visualization: bool = False

    def __post_init__(self):
        """
        StreamPreset's post init function. It will set the display_duration attribute based on the default_display_duration in the config file.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        @return:
        """
        if self.channel_names is None:  # channel names can be none when the number of channels is too big
            self.channel_names = ['c {0}'.format(i) for i in range(self.num_channels)]
            self.can_edit_channel_names = False
            print(f"StreamPreset: disabling channel editing for stream {self.stream_name}")
        if self.num_channels > config.MAX_TS_CHANNEL_NUM:
            self.can_edit_channel_names = False
            print(f"StreamPreset: disabling channel editing for stream {self.stream_name}")
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
class FMRIPreset(metaclass=SubPreset):

    stream_name: str
    preset_type: PresetType
    data_type: DataType
    num_channels: int
    data_shape: tuple[int, int, int]
    normalize: bool
    alignment: bool
    threshold: float
    nominal_sampling_rate: int = 2
    mri_file_path: str = None

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


def _load_video_device_presets():
    try:
        print('Loading available cameras')
        rtn = []
        _, working_camera_ports, _ = get_working_camera_ports()
        working_cameras_stream_names = [f'Camera {x}' for x in working_camera_ports]

        for camera_id, camera_stream_name in zip(working_camera_ports, working_cameras_stream_names):
            rtn.append(VideoPreset(camera_stream_name, PresetType.WEBCAM, camera_id))
        print("finished loading available cameras")
        return rtn
    except KeyboardInterrupt:
        print('KeyboardInterrupt: exiting')
        return []

def _load_audio_device_presets():
    try:
        print('Loading available audio devices')
        rtn = []
        # _, working_camera_ports, _ = get_working_camera_ports()
        # working_cameras_stream_names = [f'Camera {x}' for x in working_camera_ports]
        #
        # for camera_id, camera_stream_name in zip(working_camera_ports, working_cameras_stream_names):
        #     rtn.append(VideoPreset(camera_stream_name, PresetType.WEBCAM, camera_id))
        # print("finished loading available cameras")

        audio = pyaudio.PyAudio()
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')

        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                audio_preset_dict = create_default_audio_preset(
                    stream_name=audio.get_device_info_by_host_api_device_index(0, i).get('name'),
                    audio_device_index=i,
                    num_channels=audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels'),
                )
                audio_preset = AudioPreset(**audio_preset_dict)
                rtn.append(audio_preset)
                print(
                    "Create Audio Preset: ", audio_preset.stream_name,
                    ' ', audio_preset.audio_device_index,
                    ' ', audio_preset.num_channels)

        return rtn
    except KeyboardInterrupt:
        print('KeyboardInterrupt: exiting')
        return []

def create_default_audio_preset(stream_name, audio_device_index, num_channels):
    group_info = create_default_group_info(num_channels)

    audio_default_preset_dict = {
    'stream_name': stream_name,
    'audio_device_index': audio_device_index,
    'num_channels': num_channels,
    'channel_names': ['c {0}'.format(i) for i in range(num_channels)],
    'group_info': group_info,
    'device_info': {}
    }
    return audio_default_preset_dict


# def create_default_audio_preset(stream_name, audio_device_index, num_channels):
#     if is_stream_name_in_presets(stream_name):
#         raise ValueError(f'Stream preset with stream name {stream_name} already exists.')
#     audio_device_default_preset_dict = create_default_audio_preset(stream_name, audio_device_index, num_channels)
#
#     # audio_preset = AudioPreset(**audio_device_default_preset_dict)
#
#     return audio_preset



def _load_param_presets_recursive(param_preset_dict):
    rtn = []
    if isinstance(param_preset_dict, list):
        for param_preset in param_preset_dict:
            rtn.append(_load_param_presets_recursive(param_preset))
    elif isinstance(param_preset_dict, dict):
        if isinstance(param_preset_dict['value'], list):
            param_preset_dict['value'] = _load_param_presets_recursive(param_preset_dict['value'])
            return ScriptParam(**param_preset_dict)
        else:
            return ScriptParam(**param_preset_dict)
    return rtn

def save_presets_locally(app_data_path, preset_dict, file_name) -> None:
    """
    sync the presets to the local disk. This will create a _presets.json file in the app data folder if it doesn't exist.
    applies file lock while the json is being dumped. This will block another other process from accessing the file without
    raising an exception.
    """
    if not os.path.exists(app_data_path):
        os.makedirs(app_data_path)
    path = os.path.join(app_data_path, file_name)
    json_data = json.dumps(preset_dict, indent=4, cls=PresetsEncoder)
    with open(path, 'w') as f:
        f.write(json_data)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Presets(metaclass=Singleton):
    """
    dataclass containing all the presets

    Attributes:
        stream_presets: dictionary containing all the stream presets. The key is the stream name and the value is the StreamPreset object
        experiment_presets: dictionary containing all the experiment presets. The key is the experiment name and the value is a list of stream names

        _reset: if true, reload the presets. This is done in post_init by removing files at _last_mod_time_path and _preset_path.

    Note: presets must not have multiprocessing.Process in the dataclass attributes. This will cause the program to crash.
    """
    _preset_root: str = None
    _reset: bool = False

    _lsl_preset_root: str = 'LSLPresets'
    _zmq_preset_root: str = 'ZMQPresets'
    _device_preset_root: str = 'DevicePresets'
    _experiment_preset_root: str = 'ExperimentPresets'

    stream_presets: Dict[str, Union[StreamPreset, VideoPreset, AudioPreset, FMRIPreset]] = field(default_factory=dict)
    script_presets: Dict[str, ScriptPreset] = field(default_factory=dict)
    experiment_presets: Dict[str, list] = field(default_factory=dict)

    _app_data_path: str = AppConfigs().app_data_path
    _last_mod_time_path: str = os.path.join(_app_data_path, 'last_mod_times.json')
    _preset_path: str = os.path.join(_app_data_path, '_presets.json')

    def __post_init__(self):
        """
        The post init of presets does the following:
        1. if reset is true, remove the last mod time and preset json files
        2. set the private path variables based on the given preset root, which makes the preset root a mandatory argument when first time initializing _presets globally.
        3. load the presets from the local disk if it exists
        4. check if any presets are dirty and load them
        5. save the presets to the local disk
        """
        pass
        if self._preset_root is None:
            raise ValueError('preset root must not be None when first time initializing _presets')
        else:
            print(f"Preset root is set to {self._preset_root}, is exists {os.path.exists(self._preset_root)}")
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
            SplashLoadingTextNotifier().set_loading_text(f'Reloading presets from {self._app_data_path}')
            with open(self._preset_path, 'r') as f:
                preset_dict = json.load(f)
                preset_dict = {k: v for k, v in preset_dict.items() if not k.startswith('_')}  # don't load private variables
                if 'stream_presets' in preset_dict.keys():
                    for key, value in preset_dict['stream_presets'].items():
                        if PresetType.is_lsl_zmq_custom_preset(value['preset_type']):
                            preset = StreamPreset(**value)
                            preset_dict['stream_presets'][key] = preset
                        # elif PresetType.is_video_preset(value['preset_type']):  # video presets won't be loaded
                        #     preset = VideoPreset(**value)
                        #     preset_dict['stream_presets'][key] = preset
                    preset_dict['stream_presets'] = {k: v for k, v in preset_dict['stream_presets'].items() if isinstance(v, (StreamPreset, VideoPreset))}
                if 'script_presets' in preset_dict.keys():
                    for key, value in preset_dict['script_presets'].items():
                        try:
                            value['param_presets'] = [_load_param_presets_recursive(param_preset) for param_preset in value['param_presets']]
                            preset = ScriptPreset(**value)
                            preset_dict['script_presets'][key] = preset
                        except (TypeError, KeyError):
                            print(f'Script with key {key} will not be loaded, because the script preset attributes was changed during the last update')
                    preset_dict['script_presets'] = {k: v for k, v in preset_dict['script_presets'].items() if isinstance(v, ScriptPreset)}
                self.__dict__.update(preset_dict)
        dirty_presets = self._record_presets_last_modified_times()

        _load_stream_presets(self, dirty_presets)
        SplashLoadingTextNotifier().set_loading_text('Loading video devices...You may notice webcam flashing.')
        self.save(is_async=False)
        SplashLoadingTextNotifier().set_loading_text("_presets instance successfully initialized")

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

        (dirty_presets[PresetType.LSL.value], dirty_presets[PresetType.ZMQ.value], dirty_presets[
            PresetType.CUSTOM.value], dirty_presets[
             PresetType.EXPERIMENT.value]), current_mod_times = get_file_changes_multiple_dir(self._preset_roots, last_mod_times)
        if not os.path.exists(self._app_data_path):
            os.makedirs(self._app_data_path)
        with open(self._last_mod_time_path, 'w') as f:
            json.dump(current_mod_times, f)

        return dirty_presets

    def __del__(self):
        """
        save the presets to the local disk when the application is closed
        """
        save_presets_locally(self._app_data_path, self.__dict__, '_presets.json')
        print(f"_presets instance successfully deleted with its contents saved to {self._app_data_path}")
        # if self._load_video_device_process is not None and self._load_video_device_process.is_alive():
        #     self._load_video_device_process.terminate()

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

    def add_video_preset_by_fields(self, stream_name, video_type, video_id):
        video_preset = VideoPreset(stream_name, video_type, video_id)
        self.stream_presets[video_preset.stream_name] = video_preset

    def add_video_presets(self, video_presets: List[VideoPreset]):
        """
        add a list of video presets to the presets
        :param video_presets: list of video presets
        :return: None
        """
        for video_preset in video_presets:
            self.stream_presets[video_preset.stream_name] = video_preset

    def add_audio_presets(self, audio_presets: List[AudioPreset]):
        """
        add a list of audio presets to the presets
        :param audio_presets: list of audio presets
        :return: None
        """
        for audio_preset in audio_presets:
            self.stream_presets[audio_preset.stream_name] = audio_preset

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

        # if is_async:
        #     self_dict = copy.deepcopy(self.__dict__)
        #     p = multiprocessing.Process(target=save_presets_locally, args=(self._app_data_path, self_dict, '_presets.json'))
        #     p.start()
        # else:
        save_presets_locally(self._app_data_path, self.__dict__, '_presets.json')

    def __getitem__(self, key):
        return self._get_all_presets()[key]

    def keys(self):
        return self._get_all_presets().keys()

    def reload_stream_presets(self):
        """
        This function will remove the json file in AppData containing the last modified times of the presets located
        in the preset_roots. It then calls _load_stream_presets to reload the stream presets, and without the last modified
        times, all the stream presets will be reloaded.
        """
        if os.path.exists(self._last_mod_time_path):
            os.remove(self._last_mod_time_path)
        dirty_presets = self._record_presets_last_modified_times()
        _load_stream_presets(self, dirty_presets)
        self.save(is_async=True)

    # def reload_video_presets(self):
    #     """
    #     this function will start a separate process look for video devices.
    #     an outside qthread must monitor the return of this process and call _presets().add_video_presets(rtn), where
    #     rtn is the return of the process _presets()._load_video_device_process.
    #
    #
    #     """
    #     self.add_video_preset_by_fields('monitor 0', PresetType.MONITOR, 0)  # always add the monitor 0 preset
    #     _load_video_device_process = ProcessWithQueue(target=_load_video_device_presets)
    #     _load_video_device_process.start()
    #     return _load_video_device_process

    def remove_video_presets(self):
        """
        remove all the video presets
        :return: None
        """
        self.stream_presets = {stream_name: stream_preset for stream_name, stream_preset in self.stream_presets.items() if not stream_preset.preset_type.is_self_video_preset()}

    def remove_audio_presets(self):
        """
        remove all the audio presets
        :return: None
        """
        self.stream_presets = {stream_name: stream_preset for stream_name, stream_preset in self.stream_presets.items() if not stream_preset.preset_type.is_self_audio_preset()}