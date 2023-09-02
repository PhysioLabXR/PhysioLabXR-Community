from physiolabxr.presets.PresetEnums import DataType


class RenaError(Exception):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)
    """Base class for other exceptions"""
    pass




class DataPortNotOpenError(RenaError):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'attempting to read from unopened data port'

    """Raised when attempting to read from unopened data port"""
    pass


class PortsNotSetUpError(RenaError):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'CLI Ports are not set up'

    """Raised when to cli port is not set up while trying to talk to the sensor"""
    pass


class GeneralMmWError(RenaError):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'general mmWave error occurred, please debug the tlv buffer and decoder'

    pass


class BufferOverFlowError(RenaError):
    """Raised when data buffer overflows """
    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'TLV buffer overflowed'

    pass


class InterfaceNotExistError(RenaError):
    """Rasied when an interface doesn't exist when in use"""
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'Interface missing...'

    pass


class StoragePathInvalidError(RenaError):
    """Raised when the path given is invalid"""
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'Invalid path...'

    pass


class LeapPortTimeoutError(RenaError):
    """Raised when LeapInterface is running without LeapMouse"""
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'LeapMouse is not running'

    pass

class AlreadyAddedError(RenaError):
    def __init__(self):
        super().__init__()
    def __str__(self):
        return 'Failed to add. The target is already in the app \n' + self.message

    pass

class LSLStreamNotFoundError(RenaError):
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return self.message


class ChannelMismatchError(RenaError):
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return 'Opened stream has a different number of channels than in the settings \n' + self.message

class UnsupportedErrorTypeError(RenaError):
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return 'This error type should not be raised \n' + self.message

class InvalidPresetErrorChannelNameOrNumChannel(RenaError):
    def __init__(self, stream_name):
        super().__init__(stream_name)
    def __str__(self):
        return 'The preset {0} does is not valid. Must have either ChannelNames or NumChannels defined\n'.format(self.message)

class InvalidScriptPathError(RenaError):
    def __init__(self, script_path, error):
        super().__init__(error)
        self.script_path = script_path
        self.error = error

    def __str__(self):
        return 'Unable to load custom script: Invalid script path {0}. \nError: {1}\n'.format(self.script_path, self.error)

class ScriptMissingModuleError(RenaError):
    def __init__(self, script_path, error):
        super().__init__(error)
        self.script_path = script_path
        self.error = error

    def __str__(self):
        return 'Unable to load custom script: {0} \n One of the module it tries to import is missing from your python environment. \nError: {1}\n'.format(self.script_path, self.error)


class BadOutputError(RenaError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return 'Bad output. \nError: {0}\n'.format(self.error)

class ScriptSyntaxError(RenaError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return 'Script has syntax errors: ' + str(self.error) + '\n' + self.error.text

class MissingPresetError(RenaError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return 'Preset {0} is missing'.format(self.error)

class DataProcessorEvokeFailedError(RenaError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return 'DataProcessorEvokeFailedError: ' + self.error



class DaProcessorNotchFilterInvalidQError(DataProcessorEvokeFailedError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return self.error #+ 'DaProcessorNotchFilterInvalidQError'

class DataProcessorInvalidFrequencyError(DataProcessorEvokeFailedError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return self.error #+ 'DataProcessorInvalidFrequencyError'

class DataProcessorInvalidBufferSizeError(DataProcessorEvokeFailedError):
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return self.error #+ 'DataProcessorInvalidBufferSizeError'

class InvalidStreamMetaInfoError(RenaError):
    """Raised when the stream meta info is invalid"""
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return f'Invalid Stream Meta Info: {self.error}'


class ZMQPortOccupiedError(RenaError):
    """Raised when the zmq port is occupied"""
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return f'ZMQ Port Occupied: {self.error}'

class UnsupportedLSLDataTypeError(RenaError):
    """Raised when the zmq port is occupied"""
    def __init__(self, error):
        super().__init__(error)
        self.error = error

    def __str__(self):
        return f'Unsupported Data Type for LSL {self.error}. LSL only supports {DataType.get_lsl_supported_names()}'