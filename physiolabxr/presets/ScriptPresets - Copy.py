from dataclasses import dataclass
from typing import List, Union

from physiolabxr.presets.PresetEnums import PresetType, DataType
from physiolabxr.presets.preset_class_helpers import SubPreset
from physiolabxr.scripting.scripting_enums import ParamType
from physiolabxr.utils.ConfigPresetUtils import reload_enums

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ScriptParam(metaclass=SubPreset):
    name: str
    type: ParamType
    # noinspection PyTypeHints
    value: Union[tuple(ParamType.get_supported_types())]

    def __post_init__(self):
        reload_enums(self)


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ScriptOutput(metaclass=SubPreset):
    stream_name: str
    num_channels: int
    interface_type: PresetType
    data_type: DataType
    port_number: int = None

    def __post_init__(self):
        reload_enums(self)

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ScriptPreset(metaclass=SubPreset):
    id: str
    inputs: List[str]
    # outputs: List[str]
    # output_num_channels: List[int]
    # output_interfaces: List[PresetType]
    output_presets: List[ScriptOutput]
    param_presets: List[ScriptParam]

    run_frequency: int
    time_window: int
    script_path: str
    is_simulate: bool

    def __post_init__(self):
        self.output_presets = [ScriptOutput(**output) if isinstance(output, dict) else output for output in self.output_presets ]
        reload_enums(self)