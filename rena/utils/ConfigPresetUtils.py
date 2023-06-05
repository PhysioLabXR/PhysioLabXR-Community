import enum
import json
import os


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
