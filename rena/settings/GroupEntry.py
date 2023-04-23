from dataclasses import dataclass
from typing import List

from rena import config
from rena.settings.PlotConfig import PlotConfig


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class GroupEntry:
    """
    Group entry defines a group of channels to be shown in the same plot.
    Group entry is contained in the group_info dictionary of StreamPreset.
    """
    group_name: str
    channel_indices: List[int]
    is_channels_shown: List[bool] = None
    plot_config: PlotConfig = PlotConfig()

    # read-only attributes
    is_image_only: bool = None

    def __post_init__(self):
        """
        GroupEntry;s post init function. It will set the is_image_only attribute based on the number of channels in the group.
        Note any attributes loaded from the config will need to be loaded into the class's attribute here
        """
        num_channels = len(self.channel_indices)
        if self.is_channels_shown is None:  # will only modify the channel indices if it is not set. It could be set from loading a preset json file
            max_channel_shown_per_group = config.settings.value('default_channel_display_num_per_group')
            if num_channels <= max_channel_shown_per_group:
                self.is_channels_shown = [True] * num_channels
            else:
                is_channels_shown = [True] * max_channel_shown_per_group
                is_channels_shown += [False] * (num_channels - max_channel_shown_per_group)

        self.is_image_only = num_channels > config.settings.value('max_timeseries_num_channels')
