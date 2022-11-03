import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtGui import QIntValidator

from rena.ui.CustomPropertyWidget import CustomPropertyWidget
from rena.utils.settings_utils import check_preset_exists, get_stream_preset_info, get_video_device_names, \
    get_preset_category, get_stream_preset_custom_info
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

        self.device_property_fields = {}

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
        if len(self.device_property_fields) > 0:
            self.clear_custom_device_property_uis()

        stream_name = self.get_selected_stream_name()
        preset_category = get_preset_category(stream_name)
        if preset_category == 'stream':
            networking_interface = get_stream_preset_info(stream_name, "NetworkingInterface")
            data_type = get_stream_preset_info(stream_name, "DataType")
            if networking_interface == 'LSL':
                self.LSL_preset_selected()
            elif networking_interface == 'ZMQ':
                port_number = get_stream_preset_info(stream_name, "PortNumber")
                self.ZMQ_preset_selected(port_number)
            elif networking_interface == 'Device':
                self.device_preset_selected(stream_name)

            index = self.DataTypeComboBox.findText(data_type, QtCore.Qt.MatchFixedString)
            if index >= 0:
                 self.DataTypeComboBox.setCurrentIndex(index)

            else:
                self.set_data_type_to_default()
                print("Invalid data type for stream: {0} in its preset, setting data type to default".format(stream_name))
        elif preset_category == 'other':
            self.LSL_preset_selected()

        elif preset_category == 'video':
            self.hide_stream_uis()
        elif preset_category == 'exp':
            self.hide_stream_uis()
        else:
            raise Exception('Unknown preset category')

    def set_data_type_to_default(self):
        self.DataTypeComboBox.setCurrentIndex(1)

    def LSL_preset_selected(self):
        self.NetworkingInterfaceComboBox.setCurrentIndex(0)
        self.PortLineEdit.setText("")
        self.PortLineEdit.setHidden(True)

    def ZMQ_preset_selected(self, port_number):
        self.NetworkingInterfaceComboBox.show()
        self.DataTypeComboBox.show()
        self.NetworkingInterfaceComboBox.setCurrentIndex(1)
        self.PortLineEdit.setText(str(port_number))
        self.PortLineEdit.show()

    def device_preset_selected(self, device_stream_name):
        self.hide_stream_uis()
        self.add_custom_device_property_uis(device_stream_name)

    def add_custom_device_property_uis(self, device_stream_name):
        device_custom_properties = get_stream_preset_custom_info(device_stream_name)
        for property_name, property_value in device_custom_properties.items():
            custom_property_widget = CustomPropertyWidget(self, property_name, property_value)
            self.device_property_fields[property_name] = custom_property_widget
            self.horizontalLayout.insertWidget(2, custom_property_widget)

    def clear_custom_device_property_uis(self):
        for property_name, property_widget in self.device_property_fields.items():
            property_widget.deleteLater()
        self.device_property_fields = dict()
    def hide_stream_uis(self):
        self.DataTypeComboBox.setHidden(True)
        self.NetworkingInterfaceComboBox.setHidden(True)
        self.PortLineEdit.setHidden(True)
