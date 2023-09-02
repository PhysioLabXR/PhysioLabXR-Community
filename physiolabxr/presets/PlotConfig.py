from dataclasses import dataclass, asdict
from enum import Enum

from physiolabxr.presets.Cmap import Cmap
from physiolabxr.presets.preset_class_helpers import SubPreset
from physiolabxr.utils.ConfigPresetUtils import reload_enums


class ImageFormat(Enum):
    pixelmap = 0
    rgb = 1
    bgr = 2
    def depth_dim(self):
        if self == ImageFormat.pixelmap:
            return 1
        elif self == ImageFormat.rgb or self == ImageFormat.bgr:
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
    scaling_percentage: float = 100

    cmap: Cmap = Cmap.VIRIDIS

    vmin: float = None
    vmax: float = None

    vminR: float = None
    vmaxR: float = None
    vminG: float = None
    vmaxG: float = None
    vminB: float = None
    vmaxB: float = None

    def __post_init__(self):
        reload_enums(self)

    def get_valid_image_levels(self):
        """
        this function will return None if the image levels are not valid
        """
        if self.image_format == ImageFormat.pixelmap:
            if self.vmin is None or self.vmax is None:
                return None
            elif self.vmin == self.vmax:
                return None
            elif self.vmin > self.vmax:
                return None
            else:
                return (self.vmin, self.vmax)
        elif self.image_format == ImageFormat.rgb or self.image_format == ImageFormat.bgr:
            if self.vminR is None or self.vmaxR is None or self.vminG is None or self.vmaxG is None or self.vminB is None or self.vmaxB is None:
                return None
            elif self.vminR == self.vmaxR or self.vminG == self.vmaxG or self.vminB == self.vmaxB:
                return None
            elif self.vminR > self.vmaxR or self.vminG > self.vmaxG or self.vminB > self.vmaxB:
                return None
            else:
                return ((self.vminR, self.vmaxR), (self.vminG, self.vmaxG), (self.vminB, self.vmaxB))

    def get_image_levels(self):
        """
        this function will return the image levels, even if they are invalid
        """
        if self.image_format == ImageFormat.pixelmap:
            return (self.vmin, self.vmax)
        elif self.image_format == ImageFormat.rgb or self.image_format == ImageFormat.bgr:
            return ((self.vminR, self.vmaxR), (self.vminG, self.vmaxG), (self.vminB, self.vmaxB))


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
                attribute_dict = {}
                for key, value in getattr(self, attr).items():  # remove any keys that are not in the PlotConfigs class
                    if key in cls.__dict__:
                        attribute_dict[key] = value
                    else:
                        print(f'Dev Info: {key} with value {value} is not in an attribute of {attr} anymore. Possibly due to a new version of rena. Ignoring this key. Its value will be reset to default.')
                setattr(self, attr, cls(**attribute_dict))
            else:
                raise TypeError(f"Unexpected type for {attr}: {type(getattr(self, attr))}")


