from dataclasses import dataclass
from typing import List


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class GroupEntry:
    """
    Group entry defines a group of channels to be shown in the same plot.
    Group entry is contained in the group_info dictionary of StreamPreset.
    """
    group_name: str
    channel_indices: List[int]
    is_channels_shown: List[bool]
    plot_format: str