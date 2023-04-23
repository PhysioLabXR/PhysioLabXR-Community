from dataclasses import dataclass, asdict


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
    barchart_config: BarChartConfig = BarChartConfig()
    image_config: ImageConfig = ImageConfig()

    def to_dict(self):
        return {
            "barchart_config": asdict(self.barchart_config),
            "image_config": asdict(self.image_config)
        }

