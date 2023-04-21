from collections import defaultdict
from dataclasses import dataclass
from typing import List, DefaultDict

from rena.settings.GroupEntry import GroupEntry


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class StreamPreset:
    """
    Stream preset defines

    """
    # stream attributes
    stream_name: str
    channel_names: List[str]
    data_type: type

    # sampling attributes
    nominal_sampling_rate: int
    num_channels: int

    # networking attributes
    networking_interface: str
    port_number: int

    # visualization attributes
    display_duration: float

    # preprocessing attributes
    group_info: DefaultDict[str, GroupEntry]