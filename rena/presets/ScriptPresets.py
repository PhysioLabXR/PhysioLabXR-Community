from dataclasses import dataclass
from typing import List, Union

from rena.presets.preset_class_helpers import SubPreset
from rena.scripting.scripting_enums import ParamType
from rena.utils.ConfigPresetUtils import reload_enums

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ParamPreset(metaclass=SubPreset):
    name: str
    type: ParamType
    # noinspection PyTypeHints
    value: Union[tuple(ParamType.get_supported_types())]

    def __post_init__(self):
        reload_enums(self)

@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class ScriptPreset(metaclass=SubPreset):
    id: str
    inputs: List[str]
    outputs: List[str]
    output_num_channels: List[int]

    # params: List[str]
    # params_types: List[ParamType]
    # params_value_strs: List[str]

    param_presets: List[ParamPreset]

    run_frequency: int
    time_window: int
    script_path: str
    is_simulate: bool

    def __post_init__(self):
        reload_enums(self)
