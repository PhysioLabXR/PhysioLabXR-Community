from PyQt5 import QtWidgets, uic

from rena import config
from rena.utils.settings_utils import get_all_presets


class AddWidget(QtWidgets.QWidget):
    def __init__(self, ):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/AddWidget.ui", self)
        for i in get_all_presets() + config.settings.value('cameras'):
            self.add_combo_box.addItem(i)

