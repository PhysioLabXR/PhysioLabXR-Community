import enum
import json
import os

from rena.utils.dsp_utils.dsp_modules import DataProcessorType

class DeviceType(enum.Enum):
    AUDIOINPUT = 'AUDIOINPUT'
    OPENBCI = 'OPENBCI'
    TOBIIPRO = 'TOBIIPRO'
    MONITOR = 'MONITOR'
    MMWAVE = 'MMWAVE'



def save_local(app_data_path, preset_dict, file_name, encoder=None) -> None:
    """
    sync the presets to the local disk. This will create a Presets.json file in the app data folder if it doesn't exist.
    applies file lock while the json is being dumped. This will block another other process from accessing the file without
    raising an exception.
    """
    if not os.path.exists(app_data_path):
        os.makedirs(app_data_path)
    path = os.path.join(app_data_path, file_name)
    if encoder is None:
        json_data = json.dumps(preset_dict, indent=4)
    else:
        json_data = json.dumps(preset_dict, indent=4, cls=encoder)

    with open(path, 'w') as f:
        f.write(json_data)


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