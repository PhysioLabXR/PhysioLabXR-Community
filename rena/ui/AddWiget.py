import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QIntValidator

from rena.utils.settings_utils import check_preset_exists, get_stream_preset_info, get_video_device_names
from rena.utils.ui_utils import add_presets_to_combobox, update_presets_to_combobox


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
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.on_streamName_combobox_text_changed)

        self.PortLineEdit.setValidator(QIntValidator())
        self.NetworkingInterfaceComboBox.currentIndexChanged.connect(self.networking_interface_selection_changed)

        self.stream_name_combo_box.currentIndexChanged.connect(self.on_streamName_combobox_text_changed)

        self.networking_interface_selection_changed()
        self.set_data_type_to_default()

    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, pg.QtCore.Qt.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()

    def get_port_number(self):
        return self.PortLineEdit.text()

    def get_data_type(self):
        return self.DataTypeComboBox.currentText()

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

    def update_combobox_presets(self):
        update_presets_to_combobox(self.stream_name_combo_box)

    def networking_interface_selection_changed(self):
        if self.NetworkingInterfaceComboBox.currentText() == "LSL":
            self.PortLineEdit.setHidden(True)
        elif self.NetworkingInterfaceComboBox.currentText() == "ZMQ":
            self.PortLineEdit.show()

    def get_networking_interface(self):
        return self.NetworkingInterfaceComboBox.currentText()

    def on_streamName_combobox_text_changed(self):
        stream_name = self.get_selected_stream_name()
        if check_preset_exists(stream_name):
            self.NetworkingInterfaceComboBox.show()
            self.DataTypeComboBox.show()

            networking_interface = get_stream_preset_info(stream_name, "NetworkingInterface")
            data_type = get_stream_preset_info(stream_name, "DataType")
            if networking_interface == 'LSL':
                self.NetworkingInterfaceComboBox.setCurrentIndex(0)
                self.PortLineEdit.setText("")
            else:
                port_number = get_stream_preset_info(stream_name, "PortNumber")
                self.NetworkingInterfaceComboBox.setCurrentIndex(1)
                self.PortLineEdit.setText(str(port_number))

            index = self.DataTypeComboBox.findText(data_type, QtCore.Qt.MatchFixedString)
            if index >= 0:
                 self.DataTypeComboBox.setCurrentIndex(index)
            else:
                self.set_data_type_to_default()
                print("Invalid data type for stream: {0} in its preset, setting data type to default".format(stream_name))
        elif stream_name in get_video_device_names():
            self.DataTypeComboBox.setHidden(True)
            self.NetworkingInterfaceComboBox.setHidden(True)
        else:
            self.DataTypeComboBox.show()
            self.NetworkingInterfaceComboBox.show()

    def set_data_type_to_default(self):
        self.DataTypeComboBox.setCurrentIndex(1)
