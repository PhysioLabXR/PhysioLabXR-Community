# This Python file uses the following encoding: utf-8
import os

import pyqtgraph as pg
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QDialogButtonBox, QTableWidgetItem

from physiolabxr.configs import config
from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs, LinechartVizMode, RecordingFileFormat
from physiolabxr.presets.Presets import Presets, _load_video_device_presets, _load_audio_device_presets
from physiolabxr.presets.PresetEnums import PresetType, RPCLanguage
from physiolabxr.startup.startup import load_settings
from physiolabxr.threadings.ScreenCaptureWorker import get_screen_capture_size
from physiolabxr.threadings.WaitThreads import start_wait_process
from physiolabxr.ui.RPCOutputWidget import RPCOutputWidget
from physiolabxr.utils.Validators import NoCommaIntValidator
from physiolabxr.utils.ui_utils import stream_stylesheet
from physiolabxr.ui.dialogs import dialog_popup


class RPCWidget(QtWidgets.QWidget):
    def __init__(self, scripting_widget):
        super().__init__()
        self.ui = uic.loadUi(AppConfigs()._ui_RPCWidget, self)
        self.scripting_widget = scripting_widget

        self.scroll_area.setWidgetResizable(True)
        self.add_to_list_button.setIcon(AppConfigs()._icon_add)

        self.add_to_list_button.clicked.connect(self.add_to_list_button_clicked)

    def add_to_list_button_clicked(self):
        self._add_rpc_output()

    def _add_rpc_output(self, output_location=".", rpc_language=RPCLanguage.PYTHON):
        rpc_output_widget = RPCOutputWidget(self.scripting_widget, rpc_language, output_location)
        self.list_content_frame_widget.layout().insertWidget(self.list_content_frame_widget.layout().count() - 2,
                                                             rpc_output_widget)
        self.list_content_frame_widget.layout().setAlignment(rpc_output_widget, QtCore.Qt.AlignmentFlag.AlignTop)

        def remove_btn_clicked():
            self.list_content_frame_widget.layout().removeWidget(rpc_output_widget)
            rpc_output_widget.deleteLater()

        rpc_output_widget.set_remove_button_callback(remove_btn_clicked)

    def get_output_info(self):
        output_info = []
        for i in range(0, self.list_content_frame_widget.layout().count() - 2):
            output_widget = self.list_content_frame_widget.layout().itemAt(i).widget()
            if os.path.exists(output_widget.output_location_lineEdit.text()):
                # output_info.append({'language': RPCLanguage(output_widget.language_combobox.currentText()), 'location': output_widget.output_location_lineEdit.text()})
                output_info.append({'language': RPCLanguage.__members__[output_widget.language_combobox.currentText()], 'location': output_widget.output_location_lineEdit.text()})
            else:
                GlobalSignals().show_notification_signal.emit({'title': 'Invalid RPC Output Location',
                                                               'body': f"RPC Output location '{output_widget.output_location_lineEdit.text()}' does not exist. Ignored."})
        return output_info

    def check_add_default_output(self):
        if self.list_content_frame_widget.layout().count() - 2 == 0:
            GlobalSignals().show_notification_signal.emit({'title': 'No RPC Output add in RPC options',
                                                           'body': f'A default python output will be added to script directory '
                                                                   f'{os.path.dirname(self.scripting_widget.get_script_path())}. You may disable this '
                                                                   f'from the settings.'})
            self._add_rpc_output()

    def get_output_count(self):
        return self.list_content_frame_widget.layout().count() - 2

    def write_rpc_table(self, rpc_info):
        """
        Write the RPCs to the RPC table,
        each item in rpcs must be a tuple of (name, input, output, #calls, avg.run time)
        """
        self.clear_rpc_table()
        self.rpc_table_widget.setRowCount(len(rpc_info))
        for row, props in enumerate(rpc_info):
            for col, prop in enumerate(props):
                item= QTableWidgetItem(prop)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.rpc_table_widget.setItem(row, col, item)

    def clear_rpc_table(self):
        self.rpc_table_widget.setRowCount(0)

    # @QtCore.Slot()
    # def update_rpc_table(self, rpcs):
    #     self.clear_rpc_table()
    #     for rpc in rpcs:
    #         self.add_rpc_to_table(rpc)

