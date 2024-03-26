from enum import Enum

import numpy as np

try:
    from pylsl import cf_int8, cf_int16, cf_int32, cf_int64, cf_float32, cf_double64
except:
    cf_int8 = 6
    cf_int16 = 5
    cf_int32 = 4
    cf_int64 = 7
    cf_float32 = 1
    cf_double64 = 2

try:
    from pyaudio import paFloat32, paInt32, paInt24, paInt16, paInt8, paUInt8
except:
    paFloat32 = 1
    paInt32 = 2
    paInt24 = 4
    paInt16 = 8
    paInt8 = 16
    paUInt8 = 32

class CustomPresetType(Enum):
    UnicornHybridBlackBluetooth = 'UnicornHybridBlackBluetooth'


class PresetType(Enum):
    WEBCAM = 'WEBCAM'
    MONITOR = 'MONITOR'
    AUDIO = 'AUDIO'
    FMRI = 'FMRI'
    LSL = 'LSL'
    ZMQ = 'ZMQ'
    CUSTOM = 'CUSTOM'
    EXPERIMENT = 'EXPERIMENT'

    @classmethod
    def is_video_preset(cls, preset_type):
        if isinstance(preset_type, str):
            preset_type = PresetType(preset_type)
        return preset_type in [cls.WEBCAM, cls.MONITOR]

    @classmethod
    def is_lsl_zmq_custom_preset(cls, preset_type):
        if isinstance(preset_type, str):
            preset_type = PresetType(preset_type)
        return preset_type in [cls.LSL, cls.ZMQ, cls.CUSTOM]

    @classmethod
    def can_be_selected_in_gui(cls):
        return [cls.LSL, cls.ZMQ]

    def is_self_video_preset(self):
        return self in [self.WEBCAM, self.MONITOR]

    def is_self_audio_preset(self):
        return self in [self.AUDIO]



class AudioInputDataType(Enum):
    paFloat32 = paFloat32  #: 32 bit float
    paInt32 = paInt32  #: 32 bit int
    paInt24 = paInt24  #: 24 bit int
    paInt16 = paInt16  #: 16 bit int
    paInt8 = paInt8  #: 8 bit int
    paUInt8 = paUInt8  #: 8 bit unsigned int


class DataType(Enum):
    """
    Data types supported by RenaLabApp.
    Calling the enum with a value will return the same value of the  corresponding numpy data type.
    Use the class method get_data_type to get actual data type.
    """
    uint8 = "uint8"
    uint16 = "uint16"
    uint32 = "uint32"
    uint64 = "uint64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"

    def __call__(self, *args, **kwargs):
        return self.get_data_type()(args[0])

    def get_data_type(self):
        if self == DataType.uint8:
            return np.uint8
        elif self == DataType.uint16:
            return np.uint16
        elif self == DataType.uint32:
            return np.uint32
        elif self == DataType.uint64:
            return np.uint64
        elif self == DataType.int8:
            return np.int8
        elif self == DataType.int16:
            return np.int16
        elif self == DataType.int32:
            return np.int32
        elif self == DataType.int64:
            return np.int64
        elif self == DataType.float16:
            return np.float16
        elif self == DataType.float32:
            return np.float32
        elif self == DataType.float64:
            return np.float64

    def get_lsl_type(self):
        if self == DataType.int8:
            return cf_int8
        elif self == DataType.int16:
            return cf_int16
        elif self == DataType.int32:
            return cf_int32
        elif self == DataType.int64:
            return cf_int64
        elif self == DataType.float32:
            return cf_float32
        elif self == DataType.float64:
            return cf_double64
        else:
            raise ValueError(f"Data type {self} is not supported by LSL.")

    def get_struct_format(self):
        if self == DataType.uint8:
            return 'B'
        elif self == DataType.uint16:
            return 'H'
        elif self == DataType.uint32:
            return 'I'
        elif self == DataType.uint64:
            return 'Q'
        elif self == DataType.int8:
            return 'b'
        elif self == DataType.int16:
            return 'h'
        elif self == DataType.int32:
            return 'i'
        elif self == DataType.int64:
            return 'q'
        elif self == DataType.float16:
            return 'e'
        elif self == DataType.float32:
            return 'f'
        elif self == DataType.float64:
            return 'd'
        else:
            raise ValueError(f"Data type {self} is not supported by struct module.")

    @classmethod
    def get_lsl_supported_types(cls):
        return [cls.int8, cls.int16, cls.int32, cls.int64, cls.float32, cls.float64]

    @classmethod
    def get_lsl_supported_names(cls):
        return [dtype.name for dtype in cls.get_lsl_supported_types()]


class VideoDeviceChannelOrder(Enum):
    RGB = 0
    BGR = 1