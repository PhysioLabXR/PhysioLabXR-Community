from enum import Enum

import numpy as np
import pyaudio
from pylsl import cf_int8, cf_int16, cf_int32, cf_int64, cf_float32, cf_double64


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

    def is_self_video_preset(self):
        return self in [self.WEBCAM, self.MONITOR]

    def is_self_audio_preset(self):
        return self in [self.AUDIO]


class AudioInputDataType(Enum):

    paFloat32 = pyaudio.paFloat32  #: 32 bit float
    paInt32 = pyaudio.paInt32  #: 32 bit int
    paInt24 = pyaudio.paInt24  #: 24 bit int
    paInt16 = pyaudio.paInt16  #: 16 bit int
    paInt8 = pyaudio.paInt8  #: 8 bit int
    paUInt8 = pyaudio.paUInt8  #: 8 bit unsigned int


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

    @classmethod
    def get_lsl_supported_types(cls):
        return [cls.int8, cls.int16, cls.int32, cls.int64, cls.float32, cls.float64]

    @classmethod
    def get_lsl_supported_names(cls):
        return [dtype.name for dtype in cls.get_lsl_supported_types()]

class VideoDeviceChannelOrder(Enum):
    RGB = 0
    BGR = 1