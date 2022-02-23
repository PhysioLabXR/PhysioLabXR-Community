class MGesFError(Exception):
    """Base class for other exceptions"""
    pass


class DataPortNotOpenError(MGesFError):
    def __str__(self):
        return 'attempting to read from unopened data port'

    """Raised when attempting to read from unopened data port"""
    pass


class PortsNotSetUpError(MGesFError):
    def __str__(self):
        return 'CLI Ports are not set up'

    """Raised when to cli port is not set up while trying to talk to the sensor"""
    pass


class GeneralMmWError(MGesFError):
    def __str__(self):
        return 'general mmWave error occurred, please debug the tlv buffer and decoder'

    pass


class BufferOverFlowError(MGesFError):
    """Raised when data buffer overflows """

    def __str__(self):
        return 'TLV buffer overflowed'

    pass


class InterfaceNotExistError(MGesFError):
    """Rasied when an interface doesn't exist when in use"""

    def __str__(self):
        return 'Interface missing...'

    pass


class StoragePathInvalidError(MGesFError):
    """Raised when the path given is invalid"""

    def __str__(self):
        return 'Invalid path...'

    pass


class LeapPortTimeoutError(MGesFError):
    """Raised when LeapInterface is running without LeapMouse"""

    def __str__(self):
        return 'LeapMouse is not running'

    pass
