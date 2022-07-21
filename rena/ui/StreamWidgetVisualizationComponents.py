# This Python file uses the following encoding: utf-8
from copy import copy

from PyQt5 import QtWidgets, uic, sip

from rena.interfaces.LSLInletInterface import LSLInletInterface
from rena.threadings import workers
from rena.ui.OptionsWindow import OptionsWindow
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, options_icon
from rena.utils.ui_utils import AnotherWindow, dialog_popup, get_distinct_colors

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

class StreamWidgetVisualizationComponents():
    def __init__(self, fs_label, ts_label, plot_widgets, plots):
        self.fs_label=fs_label
        self.ts_label=ts_label
        self.plot_widgets=plot_widgets
        self.plots=plots

        # self.plot_group_slices=plot_group_slices
        # self.plot_formats=plot_formats

