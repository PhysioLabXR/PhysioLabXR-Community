# This Python file uses the following encoding: utf-8

from PyQt6 import uic
from PyQt6.QtWidgets import QDialog

from physiolabxr.configs import config_signal
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.utils.ui_utils import init_container, init_inputBox


class SignalSettingsTab(QDialog):
    def __init__(self, parent=None):
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        super().__init__(parent=parent)

        self.setWindowTitle('Stream Options')
        self.ui = uic.loadUi(AppConfigs()._ui_StreamSettingsTab, self)
        self.parent = parent
        # add supported filter list
        self.resize(600, 600)
        [self.SelectFilterTypeComboBox.addItem(filter_name) for filter_name in config_signal.filter_names]

        # init filter as notch
        self.filter_parameter_input_widget, self.sampling_frequency_input, self.w0_input, self.Q_input = self.init_notch_filter_inputs(
            parent=self.FilterParameterInputVerticalLayout)

        # signal connections
        self.SelectFilterTypeComboBox.currentIndexChanged.connect(self.filter_selection_changed)

        self.CancelFilterAddingBtn.clicked.connect(self.exit_filter_settings)

    def exit_filter_settings(self):
        self.close()

    def filter_selection_changed(self):
        selected_filter = str(self.SelectFilterTypeComboBox.currentText())
        print(selected_filter + ' selected')
        self.init_filter_parameter_widget(parent=self.FilterParameterInputVerticalLayout, filterType=selected_filter)

    def init_filter_parameter_widget(self, parent, filterType):

        self.filter_parameter_input_widget.deleteLater()
        if filterType == config_signal.filter_names[0]:
            self.filter_parameter_input_widget, self.sampling_frequency_input, self.w0_input, self.Q_input = self.init_notch_filter_inputs(
                parent=parent)

        if filterType == config_signal.filter_names[1]:
            self.filter_parameter_input_widget, self.sampling_frequency_input, self.fc_input, self.order_input = self.init_butterlowpass_filter_inputs(
                parent=parent)

        if filterType == config_signal.filter_names[2]:
            self.filter_parameter_input_widget, self.sampling_frequency_input, self.fc_input, self.order_input = self.init_butterhighpss_filter_inputs(
                parent=parent)

        if filterType == config_signal.filter_names[3]:
            self.filter_parameter_input_widget, self.sampling_frequency_input, self.fc_low_input, self.fc_high_input, self.order_input = self.init_butterbandpass_filter_inputs(
                parent=parent)

    def init_notch_filter_inputs(self, parent):
        filter_parameter_input_widget, filter_parameter_input_layout = init_container(parent=parent,
                                                                                      label='Notch Filter')

        sampling_frequency_input_widget, sampling_frequency_input_layout = init_container(
            parent=filter_parameter_input_layout,
            label='Sampling Frequency(Hz)',
            vertical=False)
        _, sampling_frequency_input = init_inputBox(parent=sampling_frequency_input_layout, default_input=None)

        w0_input_widget, w0_layout_input = init_container(filter_parameter_input_layout, label='Frequency to remove',
                                                          vertical=False)
        _, w0_input = init_inputBox(parent=w0_layout_input, default_input=None)

        Q_input_widget, Q_layout_input = init_container(filter_parameter_input_layout, label='Quality factor',
                                                        vertical=False)
        _, Q_input = init_inputBox(parent=Q_layout_input, default_input=None)

        return filter_parameter_input_widget, sampling_frequency_input, w0_input, Q_input

    def init_butterlowpass_filter_inputs(self, parent):
        filter_parameter_input_widget, filter_parameter_input_layout = init_container(parent=parent,
                                                                                      label='Lowpass Filter')

        sampling_frequency_input_widget, sampling_frequency_input_layout = init_container(
            parent=filter_parameter_input_layout,
            label='Sampling Frequency(Hz)',
            vertical=False)
        _, sampling_frequency_input = init_inputBox(parent=sampling_frequency_input_layout, default_input=None)

        # low pass cutoff frequency
        fc_input_widget, fc_layout_input = init_container(filter_parameter_input_layout,
                                                          label='Cutoff Frequency', vertical=False)
        _, fc_input = init_inputBox(parent=fc_layout_input, default_input=None)

        # filter order
        order_widget, order_layout_input = init_container(filter_parameter_input_layout,
                                                          label='Order', vertical=False)
        _, order_input = init_inputBox(parent=order_layout_input, default_input=None)

        return filter_parameter_input_widget, sampling_frequency_input, fc_input, order_input

    def init_butterhighpss_filter_inputs(self, parent):
        filter_parameter_input_widget, filter_parameter_input_layout = init_container(parent=parent,
                                                                                      label='High pass Filter')

        sampling_frequency_input_widget, sampling_frequency_input_layout = init_container(
            parent=filter_parameter_input_layout,
            label='Sampling Frequency(Hz)',
            vertical=False)
        _, sampling_frequency_input = init_inputBox(parent=sampling_frequency_input_layout, default_input=None)

        # high pass cutoff frequency
        fc_input_widget, fc_layout_input = init_container(filter_parameter_input_layout,
                                                          label='Cutoff Frequency', vertical=False)
        _, fc_input = init_inputBox(parent=fc_layout_input, default_input=None)

        # filter order
        order_widget, order_layout_input = init_container(filter_parameter_input_layout,
                                                          label='Order', vertical=False)
        _, order_input = init_inputBox(parent=order_layout_input, default_input=None)

        return filter_parameter_input_widget, sampling_frequency_input, fc_input, order_input

    def init_butterbandpass_filter_inputs(self, parent):
        filter_parameter_input_widget, filter_parameter_input_layout = init_container(parent=parent,
                                                                                      label='BandPass Filter')

        sampling_frequency_input_widget, sampling_frequency_input_layout = init_container(
            parent=filter_parameter_input_layout,
            label='Sampling Frequency(Hz)',
            vertical=False)
        _, sampling_frequency_input = init_inputBox(parent=sampling_frequency_input_layout, default_input=None)

        # low pass cutoff frequency
        fc_low_input_widget, fc_low_layout_input = init_container(filter_parameter_input_layout,
                                                                  label='Low Cutoff Frequency', vertical=False)
        _, fc_low_input = init_inputBox(parent=fc_low_layout_input, default_input=None)

        # high pass cutoff frequency
        fc_high_input_widget, fc_high_layout_input = init_container(filter_parameter_input_layout,
                                                                    label='High Cutoff Frequency', vertical=False)
        _, fc_high_input = init_inputBox(parent=fc_high_layout_input, default_input=None)

        # filter order
        order_widget, order_layout_input = init_container(filter_parameter_input_layout,
                                                          label='Order', vertical=False)
        _, order_input = init_inputBox(parent=order_layout_input, default_input=None)

        return filter_parameter_input_widget, sampling_frequency_input, fc_low_input, fc_high_input, order_input
