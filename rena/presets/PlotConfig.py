from dataclasses import dataclass, asdict
from enum import Enum

from rena.presets.preset_class_helpers import SubPreset

class ImageFormat(Enum):
    pixelmap = 0
    rgb = 1
    def depth_dim(self):
        if self == ImageFormat.pixelmap:
            return 1
        elif self == ImageFormat.rgb:
            return 3

class ChannelFormat(Enum):
    channel_first = 0
    channel_last = 1

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ImageConfig(metaclass=SubPreset):
    """

    """
    image_format: ImageFormat = ImageFormat.pixelmap
    width: int = 0
    height: int = 0
    channel_format: str = ChannelFormat.channel_last
    scaling: int = 1


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class BarChartConfig(metaclass=SubPreset):
    """

    """
    y_min: float = 0
    y_max: float = 1

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class TimeSeriesConfig(metaclass=SubPreset):
    """

    """
    y_min: float = 0
    y_max: float = 1

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class PlotConfigs(metaclass=SubPreset):
    """

    """
    time_series_config: TimeSeriesConfig = TimeSeriesConfig()
    barchart_config: BarChartConfig = BarChartConfig()
    image_config: ImageConfig = ImageConfig()

    def __post_init__(self):
        # recreate each of the config objects from the dict
        if isinstance(self.time_series_config, dict):
            self.time_series_config = TimeSeriesConfig(**self.time_series_config)
        if isinstance(self.barchart_config, dict):
            self.barchart_config = BarChartConfig(**self.barchart_config)
        if isinstance(self.image_config, dict):
            self.image_config = ImageConfig(**self.image_config)

    def to_dict(self):
        return {k: asdict(v) for k, v in self.__dict__.items() if not k.startswith("__")}



