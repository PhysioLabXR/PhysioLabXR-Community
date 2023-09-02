from enum import Enum


class ParamType(Enum):
    bool = bool
    int = int
    float = float
    str = str
    list = list

    @classmethod
    def get_supported_types(cls):
        return [member.value for member in list(cls)]


class ParamChange(Enum):
    ADD = 'a'
    REMOVE = 'r'
    CHANGE = 'c'
