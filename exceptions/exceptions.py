class RenaError(Exception):
    """Base class for other exceptions"""
    pass


class DataPortNotOpenError(RenaError):
    def __str__(self):
        return 'attempting to read from unopened data port'

    """Raised when attempting to read from unopened data port"""
    pass


class PortsNotSetUpError(RenaError):
    def __str__(self):
        return 'CLI Ports are not set up'

    """Raised when to cli port is not set up while trying to talk to the sensor"""
    pass


class GeneralMmWError(RenaError):
    def __str__(self):
        return 'general mmWave error occurred, please debug the tlv buffer and decoder'

    pass


class BufferOverFlowError(RenaError):
    """Raised when data buffer overflows """

    def __str__(self):
        return 'TLV buffer overflowed'

    pass


class InterfaceNotExistError(RenaError):
    """Rasied when an interface doesn't exist when in use"""

    def __str__(self):
        return 'Interface missing...'

    pass


class StoragePathInvalidError(RenaError):
    """Raised when the path given is invalid"""

    def __str__(self):
        return 'Invalid path...'

    pass


class LeapPortTimeoutError(RenaError):
    """Raised when LeapInterface is running without LeapMouse"""

    def __str__(self):
        return 'LeapMouse is not running'

    pass

class AlreadyAddedError(RenaError):
    def __str__(self):
        return 'Failed to add. The target is already in the app.'

    pass

class LSLStreamNotFoundError(RenaError):
    def __str__(self):
        return 'Unable to find to LSL stream'

    pass