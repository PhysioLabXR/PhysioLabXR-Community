from PyQt6.QtCore import QObject


class Singleton(type):
    """
    Singleton metaclass. This will ensure only one instance of the class is created.
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SingletonQObject(type(QObject), type):
    """
    Metaclass that combines QObject and Singleton functionality.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]