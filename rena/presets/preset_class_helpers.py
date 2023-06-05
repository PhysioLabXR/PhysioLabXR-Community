# from dataclasses import dataclass

from dataclasses import dataclass

from rena.utils.ConfigPresetUtils import reload_enums, DeviceType


class SubPreset(type):
    pass


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class DevicePreset(metaclass=SubPreset):
    _device_name: str
    _device_type: DeviceType
    device_nominal_sampling_rate = 4410

    def get_device_name(self):
        return self._device_name

    def get_device_type(self):
        return self._device_type

    def set_device_name(self, device_name: str):
        self._device_name = device_name

    def set_device_type(self, device_type: DeviceType):
        self._device_type = device_type

    def __post_init__(self):
        """
        VideoPreset's post init function.
        @return:
        """
        # convert any enum attribute loaded as string to the corresponding enum value
        reload_enums(self)

    # def __post_init__(self):
    #     """
    #     VideoPreset's post init function.
    #     @return:
    #     """
    #     # convert any enum attribute loaded as string to the corresponding enum value
    #     reload_enums(self)

    # @property
    # def device_type(self):
    #     return self._device_type

# @dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
# class TobiiDeviceInfo(metaclass=SubPreset, DeviceInfo)

# class Deviceinfo(type, metaclass=SubPr`):
#     pass

# def reload_enums(target):
#     for attr, attr_type in target.__annotations__.items():
#         if isinstance(attr_type, type) and issubclass(attr_type, enum.Enum):
#             value = getattr(target, attr)
#             if isinstance(value, str):
#                 setattr(target, attr, attr_type[value])
#         elif isinstance(attr_type, type) and issubclass(attr_type, list) and attr_type.__args__ and \
#                 isinstance(attr_type.__args__[0], type) and issubclass(attr_type.__args__[0], enum.Enum):
#             value = getattr(target, attr)
#             if isinstance(value, list):
#                 setattr(target, attr, [attr_type.__args__[0][v] if isinstance(v, str) else v for v in value])
