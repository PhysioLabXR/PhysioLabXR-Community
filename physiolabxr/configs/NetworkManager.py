from physiolabxr.utils.Singleton import Singleton


class NetworkManager(metaclass=Singleton):
    reserve_starting_port = 14000  # reserve ports for future use; they are used by the above services if their default ports are in use
    _reserve_ports_queue = []

    def __init__(self):
        # TODO work out the available ports
        # self._reserve_ports_queue = multiprocessing.Queue()
        # self._port_finder_process = PortFinderProcess(self._reserve_ports_queue, self.reserve_starting_port, 100)
        # self._port_finder_process.start()
        self._reserve_ports_queue = list(range(self.reserve_starting_port, self.reserve_starting_port + 100))

