# This Python file uses the following encoding: utf-8

import pyqtgraph as pg
from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtWidgets import QFileDialog, QDialogButtonBox

from physiolabxr.configs import config
from physiolabxr.configs.GlobalSignals import GlobalSignals
from physiolabxr.configs.configs import AppConfigs, LinechartVizMode, RecordingFileFormat
from physiolabxr.presets.Presets import Presets, _load_video_device_presets, _load_audio_device_presets
from physiolabxr.presets.PresetEnums import PresetType
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
        rpc_output_widget = RPCOutputWidget(self.scripting_widget)
        self.list_content_frame_widget.layout().insertWidget(self.list_content_frame_widget.layout().count() - 2, rpc_output_widget)
        self.list_content_frame_widget.layout().setAlignment(rpc_output_widget, QtCore.Qt.AlignmentFlag.AlignTop)
        def remove_btn_clicked():
            self.list_content_frame_widget.layout().removeWidget(rpc_output_widget)
            rpc_output_widget.deleteLater()
        rpc_output_widget.set_remove_button_callback(remove_btn_clicked)