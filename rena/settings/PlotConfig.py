from collections import defaultdict
from dataclasses import dataclass
from typing import List, DefaultDict

from rena import config
from rena.settings.GroupEntry import GroupEntry


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ImageConfig:
    """

    """
    image_format: str


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class PlotConfig:
    """

    """
    time_series: dict
    image_config: ImageConfig

