import threading
from abc import ABC, abstractmethod

class DataStreamInterface(ABC, threading.Thread):
    """
    This is the base class for custom data stream APIs.
    """

    """
    define variable here 
    """
    nominal_sampling_rate: float

    def __init__(self):
        super().__init__()


    @abstractmethod
    def start_stream(self):
        """
        overriding this method to define how to start the stream
        """
        pass

    @abstractmethod
    def is_stream_available(self):
        pass

    @abstractmethod
    def process_frames(self):
        pass

    @abstractmethod
    def stop_stream(self):
        pass

    def run(self) -> None:
        pass