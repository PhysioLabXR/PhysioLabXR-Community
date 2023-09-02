from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.interfaces.DataStreamInterface import DataStreamInterface
from physiolabxr.scripting.script_utils import get_target_class_name
from physiolabxr.utils.ui_utils import validate_script_path


class AddCustomDataStreamWidget(QtWidgets.QWidget):
    def __init__(self, parent, main_window):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.parent = parent
        self.main_window = main_window
        self.ui = uic.loadUi(AppConfigs()._ui_AddCustomDataStreamWidget, self)
        self.edit_variable_button.setVisible(False)

        self.locate_button.clicked.connect(self.on_locate_btn_clicked)
        self.create_button.clicked.connect(self.on_create_btn_clicked)

    def on_locate_btn_clicked(self):
        data_stream_interface_path = str(QFileDialog.getOpenFileName(self, "Select File", filter="py(*.py)")[0])
        self.process_locate_script(data_stream_interface_path)

        self.main_window

    def on_create_btn_clicked(self):
        # TODO
        pass

    def process_locate_script(self, data_stream_interface_path):
        if data_stream_interface_path != '':
            if not validate_script_path(data_stream_interface_path, DataStreamInterface):
                self.edit_variable_button.setVisible(False)
                return
            self.load_data_stream_interface_name(data_stream_interface_path)
            self.edit_variable_button.setVisible(True)
        else:
            self.edit_variable_button.setVisible(False)

    def load_data_stream_interface_name(self, data_stream_interface_path):
        self.path_line_edit.setText()
        self.info_label.setText(get_target_class_name(data_stream_interface_path, DataStreamInterface))


    def on_create_btn_clicked(self):
        pass