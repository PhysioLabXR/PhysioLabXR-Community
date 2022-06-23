# This Python file uses the following encoding: utf-8
from copy import copy

from PyQt5 import QtWidgets, uic, sip

from rena.interfaces.LSLInletInterface import LSLInletInterface
from rena.threadings import workers
from rena.ui.OptionsWindow import OptionsWindow
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, options_icon
from rena.utils.ui_utils import AnotherWindow, dialog_popup

import sys
import time
import webbrowser

import pyqtgraph as pg
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget
from PyQt5.QtWidgets import QLabel, QMessageBox
from pyqtgraph import PlotDataItem
from scipy.signal import decimate
from PyQt5 import QtCore


import numpy as np
import collections

import os

class StreamWidget(QtWidgets.QWidget):
    def __init__(self, main_parent, parent, stream_name, interface, insert_position=None):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """

        # GUI elements
        super().__init__()
        self.ui = uic.loadUi("ui/StreamContainer.ui", self)
        if type(insert_position) == int:
            parent.insertWidget(insert_position, self)
        else:
            parent.addWidget(self)
        self.parent = parent
        self.main_parent = main_parent

        self.stream_name = stream_name

        self.StreamNameLabel.setText(stream_name)
        self.set_button_icons()
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        # connect btn
        self.StartStopStreamBtn.clicked.connect(self.start_stop_stream_btn_clicked)
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)


        # data elements
        self.worker_thread = pg.QtCore.QThread(self)
        self.interface = interface
        self.lsl_worker = workers.LSLInletWorker(interface)


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

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.main_parent.lsl_workers[self.stream_name].is_streaming:
            self.main_parent.lsl_workers[self.stream_name].stop_stream()
            if not self.main_parent.lsl_workers[self.stream_name].is_streaming:
                # started
                print("sensor stopped")
                # toggle the icon
                self.StartStopStreamBtn.setText("Start Stream")
        else:
            self.main_parent.lsl_workers[self.stream_name].start_stream()
            if self.main_parent.lsl_workers[self.stream_name].is_streaming:
                # started
                print("sensor stopped")
                # toggle the icon
                self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()



    def dock_window(self):
        self.parent.insertWidget(self.parent.count() - 1,
                                                           self)
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.PopWindowBtn.setText('Pop Window')
        self.main_parent.pop_windows[self.stream_name].hide()  # tetentive measures
        self.main_parent.pop_windows.pop(self.stream_name)
        self.set_button_icons()

    def pop_window(self):
        w = AnotherWindow(self, self.remove_stream)
        self.main_parent.pop_windows[self.stream_name] = w
        w.setWindowTitle(self.stream_name)
        self.PopWindowBtn.setText('Dock Window')
        w.show()
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.dock_window)
        self.set_button_icons()



    def remove_stream(self):
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
        # stop_stream_btn.click()  # fire stop streaming first
        if self.main_parent.lsl_workers[self.stream_name].is_streaming:
            self.main_parent.lsl_workers[self.stream_name].stop_stream()
        self.worker_thread.exit()
        self.main_parent.lsl_workers.pop(self.stream_name)
        self.main_parent.worker_threads.pop(self.stream_name)
        # if this lsl connect to a device:
        if self.stream_name in self.main_parent.device_workers.keys():
            self.main_parent.device_workers[self.stream_name].stop_stream()
            self.main_parent.device_workers.pop(self.stream_name)

        self.main_parent.stream_widgets.pop(self.stream_name)
        self.parent.removeWidget(self)
        # close window if popped
        if self.stream_name in self.main_parent.pop_windows.keys():
            self.main_parent.pop_windows[self.stream_name].hide()
            # self.main_parent.pop_windows.pop(self.stream_name)
            self.deleteLater()
        else:  # use recursive delete if docked
            self.deleteLater()
        self.main_parent.LSL_data_buffer_dicts.pop(self.stream_name)
        return True

