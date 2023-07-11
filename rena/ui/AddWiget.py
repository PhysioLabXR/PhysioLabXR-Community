import pyqtgraph as pg
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import QCompleter

from rena.configs.GlobalSignals import GlobalSignals
from rena.presets.Presets import PresetType, DataType
from rena.ui.AddCustomDataStreamWidget import AddCustomDataStreamWidget
from rena.ui.CustomPropertyWidget import CustomPropertyWidget
from rena.presets.presets_utils import get_preset_category, get_stream_preset_info, get_stream_preset_custom_info
from rena.ui_shared import add_icon
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
        self.add_btn.setIcon(add_icon)

        self.add_custom_data_stream_widget = AddCustomDataStreamWidget(self, parent)
        self.layout().addWidget(self.add_custom_data_stream_widget)
        self.add_custom_data_stream_widget.setVisible(False)

        # add combobox
        GlobalSignals().stream_presets_entry_changed_signal.connect(self.on_stream_presets_entry_changed)
        add_presets_to_combobox(self.stream_name_combo_box)
        self.stream_name_combo_box.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseSensitive)
        self.connect_stream_name_combo_box_signals()

        self.PortLineEdit.setValidator(QIntValidator())

        # data type combobox
        for data_type in DataType:
            self.data_type_combo_box.addItem(data_type.value)
        self.preset_type_combobox.currentIndexChanged.connect(self.preset_type_selection_changed)
        self.set_data_type_to_default()


        self.preset_type_selection_changed()

        self.device_property_fields = {}

        self.current_selected_type = None

        self.check_can_add_input()

    def connect_stream_name_combo_box_signals(self):
        self.stream_name_combo_box.lineEdit().returnPressed.connect(self.on_streamName_comboBox_returnPressed)
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.check_can_add_input)
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.on_streamName_combobox_text_changed)
        self.stream_name_combo_box.currentIndexChanged.connect(self.on_streamName_combobox_text_changed)

    def disconnect_stream_name_combo_box_signals(self):
        self.stream_name_combo_box.lineEdit().returnPressed.disconnect(self.on_streamName_comboBox_returnPressed)
        self.stream_name_combo_box.lineEdit().textChanged.disconnect(self.check_can_add_input)
        self.stream_name_combo_box.lineEdit().textChanged.disconnect(self.on_streamName_combobox_text_changed)
        self.stream_name_combo_box.currentIndexChanged.disconnect(self.on_streamName_combobox_text_changed)

    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, Qt.MatchFlag.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()

    def get_port_number(self):
        return self.PortLineEdit.text()

    def get_data_type(self):
        return self.data_type_combo_box.currentText()

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
        if stream_name == '':
            self.add_btn.setEnabled(False)
            return
        # check for duplicate inputs
        if stream_name in self.parent.get_added_stream_names():
            self.add_btn.setEnabled(False)
        else:
            self.add_btn.setEnabled(True)

    def update_combobox_presets(self):
        update_presets_to_combobox(self.stream_name_combo_box)

    def preset_type_selection_changed(self):
        if self.preset_type_combobox.currentText() == "LSL":
            self.PortLineEdit.setHidden(True)
            self.add_custom_data_stream_widget.setVisible(False)
        elif self.preset_type_combobox.currentText() == "ZMQ":
            self.PortLineEdit.show()
            self.add_custom_data_stream_widget.setVisible(False)
        elif self.preset_type_combobox.currentText() == "Custom":
            self.PortLineEdit.setHidden(True)
            self.add_custom_data_stream_widget.setVisible(True)

    def get_selected_preset_type_str(self):
        return self.preset_type_combobox.currentText()

    def get_current_selected_type(self):
        stream_name = self.get_selected_stream_name()
        is_new_preset = False
        try:
            preset_type = get_preset_category(stream_name)
        except KeyError:
            is_new_preset = True
            preset_type = PresetType[self.get_selected_preset_type_str().upper()]
        return preset_type, is_new_preset

    def on_streamName_combobox_text_changed(self):
        if len(self.device_property_fields) > 0:
            self.clear_custom_device_property_uis()

        stream_name = self.get_selected_stream_name()
        selected_type, is_new_preset = self.get_current_selected_type()
        if is_new_preset:
            return

        if selected_type == PresetType.LSL:
            self.LSL_preset_selected(stream_name)
        elif selected_type == PresetType.ZMQ:
            port_number = get_stream_preset_info(stream_name, "port_number")
            self.ZMQ_preset_selected(stream_name, port_number)
        elif selected_type == PresetType.CUSTOM:
            self.device_preset_selected(stream_name)
        elif selected_type == PresetType.WEBCAM or selected_type == PresetType.MONITOR:
            self.hide_stream_uis()
        elif selected_type == PresetType.EXPERIMENT:
            self.hide_stream_uis()
        else: raise Exception("Unknow preset type {}".format(selected_type))

    def set_data_type_to_default(self):
        self.data_type_combo_box.setCurrentIndex(1)

    def LSL_preset_selected(self, stream_name):
        self.preset_type_combobox.setCurrentIndex(0)
        self.PortLineEdit.setText("")
        self.PortLineEdit.setHidden(True)
        self.verify_data_type(stream_name)

    def ZMQ_preset_selected(self, stream_name, port_number):
        self.preset_type_combobox.show()
        self.data_type_combo_box.show()
        self.preset_type_combobox.setCurrentIndex(1)
        self.PortLineEdit.setText(str(port_number))
        self.PortLineEdit.show()
        self.verify_data_type(stream_name)

    def device_preset_selected(self, device_stream_name):
        self.hide_stream_uis()
        self.add_custom_device_property_uis(device_stream_name)

    def add_custom_device_property_uis(self, device_stream_name):
        device_custom_properties = get_stream_preset_custom_info(device_stream_name)
        for property_name, property_value in device_custom_properties.items():
            custom_property_widget = CustomPropertyWidget(self, device_stream_name, property_name, property_value)
            self.device_property_fields[property_name] = custom_property_widget
            self.horizontalLayout.insertWidget(2, custom_property_widget)

    def clear_custom_device_property_uis(self):
        for property_name, property_widget in self.device_property_fields.items():
            property_widget.deleteLater()
        self.device_property_fields = dict()

    def hide_stream_uis(self):
        self.data_type_combo_box.setHidden(True)
        self.preset_type_combobox.setHidden(True)
        self.PortLineEdit.setHidden(True)

    def verify_data_type(self, stream_name):
        data_type_str = get_stream_preset_info(stream_name, "data_type").value
        index = self.data_type_combo_box.findText(data_type_str, Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.data_type_combo_box.setCurrentIndex(index)
        else:
            self.set_data_type_to_default()
            # print("Invalid data type for stream: {0} in its preset, setting data type to default".format(stream_name))

    def on_stream_presets_entry_changed(self):
        self.disconnect_stream_name_combo_box_signals()
        self.stream_name_combo_box.clear()
        add_presets_to_combobox(self.stream_name_combo_box)
        self.connect_stream_name_combo_box_signals()