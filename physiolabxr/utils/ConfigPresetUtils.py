import enum
import json
import os
import typing
from typing import Iterable


from physiolabxr.utils.dsp_utils.dsp_modules import DataProcessorType

class DeviceType(enum.Enum):
    AUDIOINPUT = 'AUDIOINPUT'
    OPENBCI = 'OPENBCI'
    TOBIIPRO = 'TOBIIPRO'
    MONITOR = 'MONITOR'
    MMWAVE = 'MMWAVE'





def is_iterable_type(t):
    origin = typing.get_origin(t)
    return origin is not None and issubclass(origin, Iterable)


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


def target_to_enum(target: str, enum_class: enum.Enum) -> enum.Enum:
    try:
        return enum_class[target]
    except KeyError:
        return None


# if __name__ == '__main__':
#
#     test = target_to_enum('test', DataProcessorType)

# # Example string
# enum_string = 'ButterworthLowpassFilter'
#
# # Convert string to enum
# enum_value = DataProcessorType.__members__[enum_string]
#
# if __name__ == '__main__':
#     # Example string
#     enum_string = 'ButterworthLowpassFilter'
#
#     # Convert string to enum
#     enum_value = DataProcessorType.__members__[enum_string]
#     print(enum_value)

def save_local(path, data, file_name, encoder=None):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, file_name), 'w') as f:
        json.dump(data, f, cls=encoder, indent=4)

