from dataclasses import dataclass
from enum import Enum
from typing import List

from physiolabxr.configs import config
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PlotConfig import PlotConfigs
from physiolabxr.presets.preset_class_helpers import SubPreset
from physiolabxr.utils.ConfigPresetUtils import reload_enums, target_to_enum
from dataclasses import field

from physiolabxr.utils.dsp_utils.dsp_modules import *


class PlotFormat(Enum):
    TIMESERIES = 0
    IMAGE = 1
    BARCHART = 2
    SPECTROGRAM = 3

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class GroupEntry(metaclass=SubPreset):
    """
    Group entry defines a group of channels to be shown in the same plot.
    Group entry is contained in the group_info dictionary of StreamPreset.
    """
    group_name: str
    is_channels_shown: List[bool] = None
    is_group_shown: bool = True
    plot_configs: PlotConfigs = None
    selected_plot_format: PlotFormat = None

    channel_indices: List[int] = None
    channel_indices_start_end: List[int] = None # this attribute is only used to create channel indices, for if the channel indices are too many, they won't be serialized to json
    # read-only attributes
    _is_image_only: bool = None  # this attribute is not serialized to json

    data_processors: List[DataProcessor] = field(default_factory=list)

    def __post_init__(self):
        """
        GroupEntry;s post init function. It will set the is_image_only attribute based on the number of channels in the group.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        """
        if self.channel_indices is None:
            self.channel_indices = list(range(self.channel_indices_start_end[0], self.channel_indices_start_end[1]))
        num_channels = len(self.channel_indices)
        if self.is_channels_shown is None:  # will only modify the channel indices if it is not set. It could be set from loading a preset json file
            max_channel_shown_per_group = config.settings.value('default_channel_display_num')
            if num_channels <= max_channel_shown_per_group:
                self.is_channels_shown = [True] * num_channels
            else:
                is_channels_shown = [True] * max_channel_shown_per_group
                is_channels_shown += [False] * (num_channels - max_channel_shown_per_group)

        self._is_image_only = num_channels > AppConfigs().max_timeseries_num_channels_per_group
        if self._is_image_only:
            self.selected_plot_format = PlotFormat.IMAGE
        if self.selected_plot_format is None:
            self.selected_plot_format = PlotFormat.TIMESERIES

        # recreate the PlotConfigs object from the dict
        if self.plot_configs is None:
            self.plot_configs = PlotConfigs()
        elif isinstance(self.plot_configs, dict):
            filtered_plot_configs = {}
            for key, value in self.plot_configs.items():  # remove any keys that are not in the PlotConfigs class
                if key in PlotConfigs.__dict__:
                    filtered_plot_configs[key] = value
                else:
                    print(f'Dev Info: {key} with value {value} is not in an attribute of PlotConfigs anymore. Possibly due to a new version of rena. Ignoring this key. Its value will be reset to default.')
            self.plot_configs = PlotConfigs(**filtered_plot_configs)
        else:
            raise ValueError(f'plot_configs must be a dict or None: {type(self.plot_configs)}')
        reload_enums(self)
        self.load_data_processor()

    # def to_dict(self):
    #     return {attr: value.name if attr == 'selected_plot_format' else value for attr, value in self.__dict__.items()}

    def is_image_only(self):
        return self._is_image_only

    def is_image_valid(self):
        width, height, image_format = self.plot_configs.image_config.width, self.plot_configs.image_config.height, self.plot_configs.image_config.image_format
        depth = image_format.depth_dim()

        return self.get_num_channels() == width * height * depth

    def get_num_channels(self):
        if self.channel_indices is None:
            channel_num = self.channel_indices_start_end[1] - self.channel_indices_start_end[0]
        else:
            channel_num = len(self.channel_indices)
        return channel_num

    def load_data_processor(self):
        data_processors = []
        for data_processor_dict in self.data_processors:
            data_processor_dict['data_processor_type'] = target_to_enum(data_processor_dict['data_processor_type'], DataProcessorType)
            data_processor = data_processor_lookup_table[data_processor_dict['data_processor_type']]()
            # load data processor values
            for key, value in data_processor_dict.items():
                setattr(data_processor, key, value)

            try:
                data_processor.evoke_data_processor()
                print('data processor from json evoke succeed')
            except DataProcessorEvokeFailedError as e:
                print('data processor from json evoke failed')

            data_processors.append(data_processor)

        self.data_processors = data_processors

    def reset_data_processors(self):
        for data_processor in self.data_processors:
            if data_processor.data_processor_valid:
                data_processor.set_channel_num(channel_num=len(self.channel_indices))
                data_processor.evoke_data_processor()
