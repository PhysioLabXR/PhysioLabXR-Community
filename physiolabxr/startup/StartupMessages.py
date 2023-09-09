from physiolabxr.utils.Singleton import Singleton


class StartupMessages(metaclass=Singleton):
    message_dict = dict()

    def __init__(self):
        pass

    def add_message(self, message, title):
        self.message_dict[title] = message