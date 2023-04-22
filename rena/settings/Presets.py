from collections import defaultdict
from typing import Dict, Any, List
from dataclasses import dataclass, field

from rena.settings.StreamPreset import StreamPreset


@dataclass
class Presets:
    """
    dataclass containing all the presets

    """
    stream_presets: Dict[str, StreamPreset] = field(default_factory=dict)

    def add_stream_preset(self, stream_preset_dict: Dict[str, Any]):
        """
        add a stream preset to the presets

        :param stream_preset_dict: dictionary containing the stream preset
        :return:
        """
        stream_preset = StreamPreset(**stream_preset_dict)
        self.stream_presets[stream_preset.stream_name] = stream_preset