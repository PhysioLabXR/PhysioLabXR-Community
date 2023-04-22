from collections import defaultdict
from dataclasses import dataclass
from typing import List, DefaultDict

from rena import config
from rena.settings.GroupEntry import GroupEntry


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StreamPreset:
    """
    Stream preset defines a stream to be loaded from the GUI.

    """
    stream_name: str
    channel_names: List[str]

    num_channels: int

    group_info: DefaultDict[str, GroupEntry]

    data_type: str = 'float32'

    networking_interface: str = 'LSL'
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