from dataclasses import dataclass, asdict
from enum import Enum

from rena.presets.Cmap import Cmap
from rena.presets.preset_class_helpers import SubPreset, reload_enums


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
    Note: if an attribute is a class, do not use default value otherwise it will be shared by all instances
    """
    image_format: ImageFormat = ImageFormat.pixelmap
    channel_format: ChannelFormat = ChannelFormat.channel_last

    width: int = 0
    height: int = 0
    scaling: int = 1

    def __post_init__(self):
        reload_enums(self)

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class BarChartConfig(metaclass=SubPreset):
    """
    Configuration for bar chart plot
    Attributes:
        y_min: minimum value of y axis
        y_max: maximum value of y axis
    """
    y_min: float = 0
    y_max: float = 1

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class TimeSeriesConfig(metaclass=SubPreset):
    """
    Configuration for time series plot
    """
    y_min: float = 0
    y_max: float = 1

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class SpectrogramConfig(metaclass=SubPreset):
    """
    Configuration for spectrogram plot
    """
    time_per_segment_second: float = 1/4
    time_overlap_second: float = 1/8
    cmap: Cmap = Cmap.VIRIDIS
    percentile_level_min: float = 5
    percentile_level_max: float = 95

    def __post_init__(self):
        reload_enums(self)

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class PlotConfigs(metaclass=SubPreset):
    """
    Developers are free to add more plot types and configurations. Simply add a new class with the desired configuration
    attributes.

    Note: if an attribute is a class, do not use default value otherwise it will be shared by all instances
    """
    time_series_config: TimeSeriesConfig = None
    barchart_config: BarChartConfig = None
    image_config: ImageConfig = None
    spectrogram_config: SpectrogramConfig = None

    def __post_init__(self):
        # retrieve the type annotations for the __init__ method
        init_annotations = self.__class__.__init__.__annotations__

        # iterate over the annotations and recreate each config object from the dict
        for attr, cls in init_annotations.items():
            if attr == "return":
                continue
            elif getattr(self, attr) is None:
                setattr(self, attr, cls())
            elif isinstance(getattr(self, attr), dict):
                setattr(self, attr, cls(**getattr(self, attr)))
            else:
                raise TypeError(f"Unexpected type for {attr}: {type(getattr(self, attr))}")


    # def to_dict(self):
    #     return {k: asdict(v) for k, v in self.__dict__.items() if not k.startswith("__")}



