from enum import Enum


class ParamTypes(Enum):
    bool = 'Bool'
    int = 'Int'
    float = 'Float'
    str = 'Str'
    list = 'List'


class ParamChange(Enum):
    ADD = 'a'
    REMOVE = 'r'
    CHANGE = 'c'
