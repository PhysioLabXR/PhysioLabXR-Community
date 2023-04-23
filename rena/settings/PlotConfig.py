from collections import defaultdict
from dataclasses import dataclass
from typing import List, DefaultDict

from rena import config
from rena.settings.GroupEntry import GroupEntry


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ImageConfig:
    """

    """
    image_format: str = 'pixel_map'
    width: int = 0
    height: int = 0
    channel_format: str = 'channel_last'
    scaling: int = 1


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class BarChartConfig:
    """

    """
    y_min: float = 0
    y_max: float = 1


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class PlotConfig:
    """

    """
    time_series: dict
    image_config: ImageConfig
    barchart_config: BarChartConfig
