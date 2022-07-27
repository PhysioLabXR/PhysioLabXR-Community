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


class LSLChannelMismatchError(RenaError):
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return 'Opened stream has a different number of channels than in the settings \n' + self.message

class UnsupportedErrorTypeError(RenaError):
    def __init__(self, message):
        super().__init__(message)
    def __str__(self):
        return 'This error type should not be raised \n' + self.message