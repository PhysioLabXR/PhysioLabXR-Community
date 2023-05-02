# This Python file uses the following encoding: utf-8
from typing import Dict

from rena.ui.GroupPlotWidget import GroupPlotWidget


class VizComponents():
    def __init__(self, fs_label, ts_label, plot_elements: Dict[str, GroupPlotWidget]):
        self.fs_label=fs_label
        self.ts_label=ts_label
        self.group_plots = plot_elements

