from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator

from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType, DataType, AudioInputDataType
from physiolabxr.presets.presets_utils import get_preset_type, get_stream_preset_info, get_stream_preset_custom_info, \
    change_stream_preset_port_number, change_stream_preset_type, change_stream_preset_data_type, \
    is_stream_name_in_presets, set_stream_preset_audio_device_frames_per_buffer, \
    set_stream_preset_audio_device_sampling_rate, set_stream_nominal_sampling_rate, \
    set_stream_preset_audio_device_data_type
from physiolabxr.scripting.physio.utils import string_to_enum
from physiolabxr.ui.AddCustomDataStreamWidget import AddCustomDataStreamWidget
from physiolabxr.ui.CustomPropertyWidget import CustomPropertyWidget
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import add_presets_to_combobox, update_presets_to_combobox, add_enum_values_to_combobox


class AddStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.ui = uic.loadUi(AppConfigs()._ui_AddWidget, self)
        self.add_btn.setIcon(AppConfigs()._icon_add)

        self.add_custom_data_stream_widget = AddCustomDataStreamWidget(self, parent)
        self.layout().addWidget(self.add_custom_data_stream_widget)
        self.add_custom_data_stream_widget.setVisible(False)

        # add combobox
        GlobalSignals().stream_presets_entry_changed_signal.connect(self.on_stream_presets_entry_changed)
        add_presets_to_combobox(self.stream_name_combo_box)
        self.stream_name_combo_box.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseSensitive)
        self.connect_stream_name_combo_box_signals()

        self.PortLineEdit.setValidator(NoCommaIntValidator())
        self.PortLineEdit.textChanged.connect(self.on_port_number_changed)
        self.PortLineEdit.textChanged.connect(self.check_can_add_input)

        # data type combobox
        add_enum_values_to_combobox(self.data_type_combo_box, DataType)
        add_enum_values_to_combobox(self.audio_device_data_type_combo_box, AudioInputDataType)
        self.set_audio_device_input_field_validator()

        self.data_type_combo_box.currentIndexChanged.connect(self.on_data_type_changed)
        self.set_data_type_to_default()

        add_enum_values_to_combobox(self.preset_type_combobox, PresetType)
        self.preset_type_combobox.currentIndexChanged.connect(self.preset_type_selection_changed)
        self.set_preset_type_to_default()
        # remove CUSTOM type from preset type combobox
        index = self.preset_type_combobox.findText(PresetType.CUSTOM.value, Qt.MatchFlag.MatchFixedString)
        self.preset_type_combobox.removeItem(index)

        self.update_preset_type_uis()
        self.device_property_fields = {}
        self.current_selected_type = None
        self.check_can_add_input()

    def connect_stream_name_combo_box_signals(self):
        self.stream_name_combo_box.lineEdit().returnPressed.connect(self.on_streamName_comboBox_returnPressed)
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.check_can_add_input)
        self.stream_name_combo_box.lineEdit().textChanged.connect(self.on_streamName_combobox_text_changed)
        self.stream_name_combo_box.currentIndexChanged.connect(self.on_streamName_combobox_text_changed)
        self.connect_audio_input_settings_on_change_signals()

    def disconnect_stream_name_combo_box_signals(self):
        self.stream_name_combo_box.lineEdit().returnPressed.disconnect(self.on_streamName_comboBox_returnPressed)
        self.stream_name_combo_box.lineEdit().textChanged.disconnect(self.check_can_add_input)
        self.stream_name_combo_box.lineEdit().textChanged.disconnect(self.on_streamName_combobox_text_changed)
        self.stream_name_combo_box.currentIndexChanged.disconnect(self.on_streamName_combobox_text_changed)
        self.disconnect_audio_input_settings_on_change_signals()


    def connect_audio_input_settings_on_change_signals(self):
        self.audio_device_sampling_rate_line_edit.textChanged.connect(self.on_audio_device_sampling_rate_line_edit_changed)
        self.audio_device_frames_per_buffer_line_edit.textChanged.connect(self.on_audio_device_frames_per_buffer_line_edit_changed)
        self.audio_device_data_type_combo_box.currentIndexChanged.connect(self.on_audio_input_data_type_combobox_changed)

    def disconnect_audio_input_settings_on_change_signals(self):
        self.audio_device_sampling_rate_line_edit.textChanged.disconnect(self.on_audio_device_sampling_rate_line_edit_changed)
        self.audio_device_frames_per_buffer_line_edit.textChanged.disconnect(self.on_audio_device_frames_per_buffer_line_edit_changed)
        self.audio_device_data_type_combo_box.currentIndexChanged.disconnect(self.on_audio_input_data_type_combobox_changed)


    def select_by_stream_name(self, stream_name):
        index = self.stream_name_combo_box.findText(stream_name, Qt.MatchFlag.MatchFixedString)
        self.stream_name_combo_box.setCurrentIndex(index)

    def get_selected_stream_name(self):
        return self.stream_name_combo_box.currentText()

    def get_selected_stream_name_is_new(self):
        stream_name = self.stream_name_combo_box.currentText()
        return stream_name, not is_stream_name_in_presets(stream_name)

    def get_data_type(self):
        return DataType(self.data_type_combo_box.currentText())

    def set_selection_text(self, stream_name):
        self.stream_name_combo_box.lineEdit().setText(stream_name)

    def on_streamName_comboBox_returnPressed(self):
        print('Enter pressed in add widget combo box with text: {}'.format(self.get_selected_stream_name()))
        self.add_btn.click()

    def check_can_add_input(self):
        """
        Caller to this function must edit the meta info in the preset if needed.
        will disable the add button if duplicate input exists
        """
        stream_name = self.stream_name_combo_box.currentText()
        stream_type = self.get_selected_preset_type()
        if stream_name == '':
            self.add_btn.setEnabled(False)
            return
        # check for duplicate inputs
        can_add = True
        can_add = can_add and (stream_name not in self.parent.get_added_stream_names())
        if stream_type == PresetType.ZMQ: can_add = can_add and self.get_port_number() != -1

        self.add_btn.setEnabled(can_add)

    def update_combobox_presets(self):
        update_presets_to_combobox(self.stream_name_combo_box)

    def preset_type_selection_changed(self):
        self.update_preset_type_uis()
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        current_type = PresetType(self.preset_type_combobox.currentText().upper())
        if not is_new: change_stream_preset_type(stream_name, current_type)  # update the preset type in the preset singleton
        self.check_can_add_input()

    def update_preset_type_uis(self):
        current_type = PresetType(self.preset_type_combobox.currentText().upper())
        if current_type == PresetType.LSL:
            self.PortLineEdit.setHidden(True)
            self.data_type_combo_box.setHidden(True)
            self.audio_device_settings_widget.setHidden(True)
            self.add_custom_data_stream_widget.setVisible(False)
        elif current_type == PresetType.ZMQ:
            self.PortLineEdit.show()
            self.data_type_combo_box.setHidden(False)
            self.audio_device_settings_widget.setHidden(True)
            self.add_custom_data_stream_widget.setVisible(False)
        elif current_type == PresetType.CUSTOM:
            self.PortLineEdit.setHidden(True)
            self.data_type_combo_box.setHidden(False)
            self.audio_device_settings_widget.setHidden(True)
            self.add_custom_data_stream_widget.setVisible(True)

    def get_selected_preset_type_str(self):
        return self.preset_type_combobox.currentText()

    def get_selected_preset_type(self):
        return PresetType[self.get_selected_preset_type_str().upper()]

    def get_selected_stream_type_in_preset(self):
        stream_name = self.get_selected_stream_name()
        is_new_preset = False
        try:
            preset_type = get_preset_type(stream_name)
        except KeyError:
            is_new_preset = True
            preset_type = PresetType[self.get_selected_preset_type_str().upper()]
        return preset_type, is_new_preset

    def on_streamName_combobox_text_changed(self):
        """
        when stream name changes, check if the stream name is a new stream name

        if it is new, it's preset type is LSL

        if it is not new and it is a stream preset, we need to populate the UI with its data type and port number.

        We also need to change the preset type to what's in the preset if it is an existing preset,
        otherwise we change the _ui based on whatever type is currently in the preset type combobox.
        @return:
        """
        if len(self.device_property_fields) > 0:
            self.clear_custom_device_property_uis()

        stream_name = self.get_selected_stream_name()
        selected_type, is_new_preset = self.get_selected_stream_type_in_preset()

        if PresetType.is_lsl_zmq_custom_preset(selected_type) and not is_new_preset:
            self.load_port_num_into_ui(stream_name)
            self.load_data_type_into_ui(stream_name)

        if PresetType.is_self_audio_preset(selected_type) and not is_new_preset:
            self.load_audio_device_settings_into_ui(stream_name)

        if is_new_preset:
            selected_type = PresetType.LSL

        preset_type_combobox_index = self.preset_type_combobox.findText(selected_type.value)
        self.preset_type_combobox.setCurrentIndex(preset_type_combobox_index)

        if selected_type == PresetType.LSL:
            self.show_lsl_preset_ui()
        elif selected_type == PresetType.ZMQ:
            self.show_zmq_preset_ui()
        elif selected_type == PresetType.CUSTOM:
            self.show_custom_preset_ui(stream_name)
        elif selected_type == PresetType.WEBCAM or selected_type == PresetType.MONITOR:
            self.show_video_uis(selected_type)
        elif selected_type == PresetType.AUDIO:
            self.show_audio_uis(selected_type)
        elif selected_type == PresetType.EXPERIMENT:
            self.hide_stream_uis()
        elif selected_type == PresetType.FMRI:
            self.hide_stream_uis()
        else: raise Exception("Unknow preset type {}".format(selected_type))

    def set_data_type_to_default(self):
        index = self.data_type_combo_box.findText(DataType.float32.value, Qt.MatchFlag.MatchFixedString)
        self.data_type_combo_box.setCurrentIndex(index)

    def set_preset_type_to_default(self):
        index = self.preset_type_combobox.findText(PresetType.LSL.value, Qt.MatchFlag.MatchFixedString)
        self.preset_type_combobox.setCurrentIndex(index)

    def show_lsl_preset_ui(self):
        self.preset_type_combobox.show()
        self.data_type_combo_box.hide()
        self.PortLineEdit.setHidden(True)
        self.preset_type_combobox.setEnabled(True)

    def show_zmq_preset_ui(self):
        self.preset_type_combobox.show()
        self.data_type_combo_box.show()
        self.PortLineEdit.show()
        self.preset_type_combobox.setEnabled(True)

    def show_custom_preset_ui(self, custom_stream_name):
        self.add_custom_device_property_uis(custom_stream_name)
        self.PortLineEdit.setHidden(True)
        self.data_type_combo_box.show()
        self.preset_type_combobox.setEnabled(True)

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

    def show_video_uis(self, selected_type):
        self.PortLineEdit.setHidden(True)
        self.data_type_combo_box.setHidden(True)
        self.preset_type_combobox.setEnabled(False)
        self.audio_device_settings_widget.setHidden(True)

    def show_audio_uis(self, selected_type):
        self.PortLineEdit.setHidden(True)
        self.data_type_combo_box.setHidden(True)
        self.preset_type_combobox.setEnabled(False)
        self.audio_device_settings_widget.setHidden(False)



    def hide_stream_uis(self):
        self.data_type_combo_box.setHidden(True)
        self.PortLineEdit.setHidden(True)
        self.audio_device_settings_widget.setHidden(True)
        self.preset_type_combobox.setEnabled(False)

    def load_data_type_into_ui(self, stream_name):
        data_type_str = get_stream_preset_info(stream_name, "data_type").value
        index = self.data_type_combo_box.findText(data_type_str, Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.data_type_combo_box.setCurrentIndex(index)
        else:
            self.set_data_type_to_default()
            # print("Invalid data type for stream: {0} in its preset, setting data type to default".format(stream_name))

    def load_port_num_into_ui(self, stream_name):
        port_number = get_stream_preset_info(stream_name, "port_number")
        if port_number is not None: self.PortLineEdit.setText(str(port_number))


    # audio_device_data_format: int = pyaudio.paInt16
    # audio_device_frames_per_buffer: int = 128
    # audio_device_sampling_rate: float = 4000
    def load_audio_device_settings_into_ui(self, stream_name):
        self.audio_device_frames_per_buffer_line_edit.setText(str(get_stream_preset_info(stream_name, "audio_device_frames_per_buffer")))
        self.audio_device_sampling_rate_line_edit.setText(str(get_stream_preset_info(stream_name, "audio_device_sampling_rate")))
        self.audio_device_data_type_combo_box.setCurrentText(get_stream_preset_info(stream_name, "audio_device_data_format").name)

    def on_stream_presets_entry_changed(self):
        self.disconnect_stream_name_combo_box_signals()
        self.stream_name_combo_box.clear()
        add_presets_to_combobox(self.stream_name_combo_box)
        self.connect_stream_name_combo_box_signals()

    def on_port_number_changed(self):
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        port_number = self.get_port_number()
        if port_number != -1 and not is_new:
            change_stream_preset_port_number(self.get_selected_stream_name(), port_number)

    def get_port_number(self):
        try:
            return int(self.PortLineEdit.text())
        except ValueError:
            return -1

    def on_data_type_changed(self):
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        if not is_new:
            data_type = DataType(self.data_type_combo_box.currentText())
            change_stream_preset_data_type(self.get_selected_stream_name(), data_type)

    def set_audio_device_input_field_validator(self):
        self.audio_device_frames_per_buffer_line_edit.setValidator(QIntValidator(0, 2048))
        self.audio_device_sampling_rate_line_edit.setValidator(QIntValidator(0, 32768))



########################################################

    def on_audio_device_frames_per_buffer_line_edit_changed(self):
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        if not is_new:
            frames_per_buffer = self.get_audio_device_frames_per_buffer()
            set_stream_preset_audio_device_frames_per_buffer(stream_name, frames_per_buffer)

    def get_audio_device_frames_per_buffer(self):
        try:
            return int(self.audio_device_frames_per_buffer_line_edit.text())
        except ValueError:
            return -1

    def on_audio_device_sampling_rate_line_edit_changed(self):
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        if not is_new:
            sampling_rate = self.get_audio_device_sampling_rate()
            set_stream_preset_audio_device_sampling_rate(stream_name, sampling_rate)
            set_stream_nominal_sampling_rate(stream_name, sampling_rate)


    def get_audio_device_sampling_rate(self):
        try:
            return int(self.audio_device_sampling_rate_line_edit.text())
        except ValueError:
            return -1

    def on_audio_input_data_type_combobox_changed(self):
        stream_name, is_new = self.get_selected_stream_name_is_new()
        if stream_name == '':
            return
        if not is_new:
            data_type = self.get_audio_input_data_type()
            set_stream_preset_audio_device_data_type(stream_name, data_type)

    def get_audio_input_data_type(self):
        return string_to_enum(enum_type=AudioInputDataType, string_value=self.audio_device_data_type_combo_box.currentText())






