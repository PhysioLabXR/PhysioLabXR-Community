# This Python file uses the following encoding: utf-8
from typing import Dict

from physiolabxr.ui.GroupPlotWidget import GroupPlotWidget


class VizComponents():
    def __init__(self, fs_label, ts_label, plot_elements: Dict[str, GroupPlotWidget]):
        self.fs_label=fs_label
        self.ts_label=ts_label
        self.group_plots = plot_elements

    def update_nominal_sampling_rate(self):
        for group_plot in self.group_plots.values():
            group_plot.update_nominal_sampling_rate()

    def set_spectrogram_cmap(self, group_name):
        self.group_plots[group_name].set_spectrogram_cmap()
