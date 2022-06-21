# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets, uic

from rena.ui.OptionsWindow import OptionsWindow
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, options_icon


class StreamWidget(QtWidgets.QWidget):
    def __init__(self, parent, stream_name, insert_position=None):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__()
        self.ui = uic.loadUi("ui/StreamContainer.ui", self)
        if type(insert_position) == int:
            parent.insertWidget(insert_position, self)
        else:
            parent.addWidget(self)

        self.stream_name = stream_name

        self.StreamNameLabel.setText(stream_name)
        self.set_button_icons()
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        # connect btn
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)


    def set_button_icons(self):
        if 'Start' in self.StartStopStreamBtn.text():
            self.StartStopStreamBtn.setIcon(start_stream_icon)
        else:
            self.StartStopStreamBtn.setIcon(stop_stream_icon)

        if 'Pop' in self.PopWindowBtn.text():
            self.PopWindowBtn.setIcon(pop_window_icon)
        else:
            self.PopWindowBtn.setIcon(dock_window_icon)


    def options_btn_clicked(self):

        print("Option window open")
        signal_settings_window = OptionsWindow(parent=self)
        if signal_settings_window.exec_():
            print("signal setting window open")
        else:
            print("Cancel!")
