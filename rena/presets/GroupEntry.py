from dataclasses import dataclass
from enum import Enum
from typing import List

from rena import config
from rena.presets.PlotConfig import PlotConfigs
from rena.presets.preset_class_helpers import reload_enums, SubPreset


class PlotFormat(Enum):
    TIMESERIES = 0
    BARCHART = 1
    IMAGE = 2

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class GroupEntry(metaclass=SubPreset):
    """
    Group entry defines a group of channels to be shown in the same plot.
    Group entry is contained in the group_info dictionary of StreamPreset.
    """
    group_name: str
    channel_indices: List[int]
    is_channels_shown: List[bool] = None
    is_group_shown: bool = True
    plot_configs: PlotConfigs = PlotConfigs()
    selected_plot_format: PlotFormat = None

    # read-only attributes
    _is_image_only: bool = None  # this attribute is not serialized to json

    def __post_init__(self):
        """
        GroupEntry;s post init function. It will set the is_image_only attribute based on the number of channels in the group.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        """
        num_channels = len(self.channel_indices)
        if self.is_channels_shown is None:  # will only modify the channel indices if it is not set. It could be set from loading a preset json file
            max_channel_shown_per_group = config.settings.value('default_channel_display_num')
            if num_channels <= max_channel_shown_per_group:
                self.is_channels_shown = [True] * num_channels
            else:
                is_channels_shown = [True] * max_channel_shown_per_group
                is_channels_shown += [False] * (num_channels - max_channel_shown_per_group)

        self._is_image_only = num_channels > config.settings.value('max_timeseries_num_channels')
        if self._is_image_only:
            self.selected_plot_format = PlotFormat.IMAGE
        if self.selected_plot_format is None:
            self.selected_plot_format = PlotFormat.TIMESERIES

        # recreate the PlotConfigs object from the dict
        if isinstance(self.plot_configs, dict):
            self.plot_configs = PlotConfigs(**self.plot_configs)
        reload_enums(self)

    # def to_dict(self):
    #     return {attr: value.name if attr == 'selected_plot_format' else value for attr, value in self.__dict__.items()}

    def is_image_only(self):
        return self._is_image_only

    def is_image_valid(self):
        width, height, image_format = self.plot_configs.image_config.width, self.plot_configs.image_config.height, self.plot_configs.image_config.image_format
        depth = image_format.depth_dim()
        return len(self.channel_indices) == width * height * depth
