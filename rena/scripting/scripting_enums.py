from enum import Enum


class ParamType(Enum):
    bool = bool
    int = int
    float = float
    str = str
    list = list


class ParamChange(Enum):
    ADD = 'a'
    REMOVE = 'r'
    CHANGE = 'c'
