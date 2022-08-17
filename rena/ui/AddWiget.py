import pyqtgraph as pg
from PyQt5 import QtWidgets, uic

from rena.utils.ui_utils import add_presets_to_combobox


class AddStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi("ui/AddWidget.ui", self)
        add_presets_to_combobox(self.stream_name_combo_box)
        self.stream_name_combo_box.lineEdit().returnPressed.connect(self.on_streamName_comboBox_returnPressed)
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.check_can_add_input)

    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()

    def set_selection_text(self, stream_name):
        self.stream_name_combo_box.lineEdit().setText(stream_name)

    def on_streamName_comboBox_returnPressed(self):
        print('Enter pressed in add widget combo box with text: {}'.format(self.get_selected_stream_name()))
        self.add_btn.click()

    def check_can_add_input(self):
        """
        will disable the add button if duplicate input exists
        """
        stream_name = self.stream_name_combo_box.currentText()
        if stream_name in self.parent.get_added_stream_names():
            self.add_btn.setEnabled(False)
        else:
            self.add_btn.setEnabled(True)
