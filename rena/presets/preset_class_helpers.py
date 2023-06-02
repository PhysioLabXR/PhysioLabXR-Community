import enum
import json
from enum import Enum

class SubPreset(type):
    pass

class DevicePreset(metaclass=SubPreset):
    pass

# @dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
# class TobiiDeviceInfo(metaclass=SubPreset, DeviceInfo)

# class Deviceinfo(type, metaclass=SubPr`):
#     pass

def reload_enums(target):
    for attr, attr_type in target.__annotations__.items():
        if isinstance(attr_type, type) and issubclass(attr_type, enum.Enum):
            value = getattr(target, attr)
            if isinstance(value, str):
                setattr(target, attr, attr_type[value])
        elif isinstance(attr_type, type) and issubclass(attr_type, list) and attr_type.__args__ and \
                isinstance(attr_type.__args__[0], type) and issubclass(attr_type.__args__[0], enum.Enum):
            value = getattr(target, attr)
            if isinstance(value, list):
                setattr(target, attr, [attr_type.__args__[0][v] if isinstance(v, str) else v for v in value])




