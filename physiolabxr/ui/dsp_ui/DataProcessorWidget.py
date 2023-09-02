# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets
from PyQt6 import uic
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator, QDoubleValidator
from PyQt6.QtGui import QPixmap

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.presets_utils import add_data_processor_to_group_entry, remove_data_processor_to_group_entry, \
    get_group_channel_num, get_group_data_processors, get_stream_nominal_sampling_rate
from physiolabxr.utils.Validators import NoCommaIntValidator
# from physiolabxr._ui.dsp_ui.OptionsWindowDataProcessingWidget import OptionsWindowDataProcessingWidget
from physiolabxr.utils.dsp_utils.dsp_modules import *


class DataProcessorWidget(QtWidgets.QWidget):

    def __init__(self, parent, data_processor: DataProcessor, adding_data_processor=False):
        super().__init__()
        self.parent = parent
        self.data_processor_invalid_pixmap = QPixmap(AppConfigs()._stream_unavailable)
        self.data_processor_valid_pixmap = QPixmap(AppConfigs()._stream_available)
        self.data_processor_activated_pixmap = QPixmap(AppConfigs()._stream_viz_active)
        self.data_processor = data_processor

        if adding_data_processor:
            self.add_data_processor_to_group_entry()

    def __post_init__(self):
        self.removeDataProcessorBtn.clicked.connect(self.remove_data_processor_btn_clicked)
        self.removeDataProcessorBtn.setIcon(AppConfigs()._icon_minus)

        self.set_data_processor_input_field_value()
        self.set_data_processor_input_field_constrain()
        self.connect_data_processor_input_field_signal()
        self.data_processor_settings_on_changed() # try to evoke data processor at the first place
        # self.set_data_processor_state_label()

        # self.data_processor.data_processor_valid_signal.connect(self.set_data_processor_state_label)
        # self.data_processor.data_processor_activated_signal.connect(self.set_data_processor_state_label)

    def add_data_processor_to_group_entry(self):
        # add data processor to group
        add_data_processor_to_group_entry(self.parent.stream_name,
                                          self.parent.group_name,
                                          data_processor=self.data_processor)
        self.data_processor.set_channel_num(channel_num=get_group_channel_num(self.parent.stream_name,
                                                                              self.parent.group_name))

    def data_processor_group_channels_on_change(self, channel_num):
        self.data_processor.set_channel_num(channel_num=channel_num)
        self.evoke_data_processor()

    def remove_data_processor_btn_clicked(self):
        # remove data processor from the group
        remove_data_processor_to_group_entry(self.parent.stream_name,
                                             self.parent.group_name,
                                             data_processor=self.data_processor)

        # remove the widget
        self.parent.remove_data_processor_widget(self)

    def set_data_processor_input_field_value(self):
        state = Qt.CheckState.Checked if self.data_processor.data_processor_activated else Qt.CheckState.Unchecked
        self.ActivateDataProcessorCheckbox.setCheckState(state)

    def set_data_processor_input_field_constrain(self):
        pass

    def connect_data_processor_input_field_signal(self):
        # connect activate signal
        self.ActivateDataProcessorCheckbox.stateChanged.connect(self.activate_data_processor_checkbox_on_changed)

    def activate_data_processor_checkbox_on_changed(self):
        check_state = self.ActivateDataProcessorCheckbox.checkState()

        if check_state == Qt.CheckState.Checked and not self.data_processor.data_processor_activated:
            self.data_processor.set_data_processor_activated(True)
        else:
            self.data_processor.set_data_processor_activated(False)
        self.set_data_processor_state_label()

    def data_processor_settings_on_changed(self):
        self.set_data_processor_params()
        self.evoke_data_processor()
        self.set_data_processor_state_label()

    def evoke_data_processor(self):
        try:
            self.data_processor.evoke_data_processor()
            # set message box text
            self.DataProcessorEvokeMessageWidget.hide()
            self.DataProcessorEvokeMessageLabel.setText('')
            self.DataProcessorEvokeMessageLabel.setStyleSheet('color: green')
        except DataProcessorEvokeFailedError as e:
            self.DataProcessorEvokeMessageWidget.show()
            self.DataProcessorEvokeMessageLabel.setText(str(e))
            self.DataProcessorEvokeMessageLabel.setStyleSheet('color: red')
            print(str(e))

    def set_data_processor_params(self):
        pass

    def set_data_processor_state_label(self):
        print(self.data_processor.data_processor_valid, self.data_processor.data_processor_activated)
        if not self.data_processor.data_processor_valid:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_invalid_pixmap)
        elif self.data_processor.data_processor_valid and self.data_processor.data_processor_activated:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_activated_pixmap)
        elif self.data_processor.data_processor_valid:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_valid_pixmap)
        else:
            self.DataProcessorStateLabel.setPixmap(self.data_processor_invalid_pixmap)

    def init_data_processor(self):
        # create corresponding data processor if data processor is None and fill with the default parameters
        pass


class NotchFilterWidget(DataProcessorWidget):

    def __init__(self, parent, data_processor=None,
                 adding_data_processor=False):
        if data_processor is None:
            data_processor = NotchFilter()
            data_processor.fs = float(get_stream_nominal_sampling_rate(parent.stream_name))


        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_NotchFilterWidget, self)
        # self.data_processor = data_processor

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(NotchFilterWidget, self).set_data_processor_input_field_value()

        self.w0LineEdit.setText(str(self.data_processor.w0))
        self.QLineEdit.setText(str(self.data_processor.Q))
        self.fsLineEdit.setText(str(self.data_processor.fs))

    def set_data_processor_input_field_constrain(self):
        self.w0LineEdit.setValidator(QDoubleValidator())
        self.QLineEdit.setValidator(QDoubleValidator())
        self.fsLineEdit.setValidator(QDoubleValidator())

    def connect_data_processor_input_field_signal(self):
        super(NotchFilterWidget, self).connect_data_processor_input_field_signal()

        self.w0LineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.QLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        w0 = self.get_w0()
        Q = self.get_Q()
        fs = self.get_fs()

        self.data_processor.set_data_processor_params(w0=w0, Q=Q, fs=fs)

    def get_w0(self):
        try:
            w0_value = float(self.w0LineEdit.text())
        except ValueError:
            return 0
        return w0_value

    def get_Q(self):
        try:
            Q_value = float(self.QLineEdit.text())
        except ValueError:
            return 0
        return Q_value

    def get_fs(self):
        try:
            fs = abs(float(self.fsLineEdit.text()))
        except ValueError:
            return 0
        return fs


class ButterworthBandPassFilterWidget(DataProcessorWidget):

    def __init__(self, parent, data_processor=None,
                 adding_data_processor=False):
        if data_processor is None:
            data_processor = ButterworthBandpassFilter()
            data_processor.fs = float(get_stream_nominal_sampling_rate(parent.stream_name))

        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_ButterworthBandPassFilterWidget, self)
        # self.data_processor = data_processor

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(ButterworthBandPassFilterWidget, self).set_data_processor_input_field_value()

        self.lowcutLineEdit.setText(str(self.data_processor.lowcut))
        self.highcutLineEdit.setText(str(self.data_processor.highcut))
        self.fsLineEdit.setText(str(self.data_processor.fs))
        self.orderLineEdit.setText(str(self.data_processor.order))

    def set_data_processor_input_field_constrain(self):
        self.lowcutLineEdit.setValidator(QDoubleValidator())
        self.highcutLineEdit.setValidator(QDoubleValidator())
        self.fsLineEdit.setValidator(QDoubleValidator())
        self.orderLineEdit.setValidator(NoCommaIntValidator())

    def connect_data_processor_input_field_signal(self):
        super(ButterworthBandPassFilterWidget, self).connect_data_processor_input_field_signal()
        self.lowcutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.highcutLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.orderLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        lowcut = self.get_lowcut()
        highcut = self.get_highcut()
        fs = self.get_fs()
        order = self.get_order()

        self.data_processor.set_data_processor_params(lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    def get_lowcut(self):
        try:
            lowcut = abs(float(self.lowcutLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return lowcut

    def get_highcut(self):
        try:
            highcut = abs(float(self.highcutLineEdit.text()))
        except ValueError:
            return 0
        return highcut

    def get_fs(self):
        try:
            fs = abs(float(self.fsLineEdit.text()))
        except ValueError:
            return 0
        return fs

    def get_order(self):
        try:
            order = abs(int(self.orderLineEdit.text()))
        except ValueError:
            return 0
        return order


class ButterworthLowpassFilterWidget(DataProcessorWidget):
    def __init__(self, parent, data_processor=None, adding_data_processor=False):
        if data_processor is None:
            data_processor = ButterworthLowpassFilter()
            data_processor.fs = float(get_stream_nominal_sampling_rate(parent.stream_name))


        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_ButterworthLowPassFilterWidget, self)
        # self.data_processor = data_processor

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(ButterworthLowpassFilterWidget, self).set_data_processor_input_field_value()

        self.cutoffLineEdit.setText(str(self.data_processor.cutoff))
        self.fsLineEdit.setText(str(self.data_processor.fs))
        self.orderLineEdit.setText(str(self.data_processor.order))

    def set_data_processor_input_field_constrain(self):
        self.cutoffLineEdit.setValidator(QDoubleValidator())
        self.fsLineEdit.setValidator(QDoubleValidator())
        self.orderLineEdit.setValidator(NoCommaIntValidator())

    def connect_data_processor_input_field_signal(self):
        super(ButterworthLowpassFilterWidget, self).connect_data_processor_input_field_signal()
        self.cutoffLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.orderLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        cutoff = self.get_cutoff()
        fs = self.get_fs()
        order = self.get_order()

        self.data_processor.set_data_processor_params(cutoff=cutoff, fs=fs, order=order)

    def get_cutoff(self):
        try:
            cutoff = abs(float(self.cutoffLineEdit.text()))
        except ValueError:
            return 0
        return cutoff

    def get_fs(self):
        try:
            fs = abs(float(self.fsLineEdit.text()))
        except ValueError:
            return 0
        return fs

    def get_order(self):
        try:
            order = abs(int(self.orderLineEdit.text()))
        except ValueError:
            return 0
        return order


class ButterworthHighpassFilterWidget(DataProcessorWidget):
    def __init__(self, parent, data_processor=None, adding_data_processor=False):
        if data_processor is None:
            data_processor = ButterworthHighpassFilter()
            data_processor.fs = float(get_stream_nominal_sampling_rate(parent.stream_name))


        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_ButterworthHighPassFilterWidget, self)
        # self.data_processor = data_processor

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(ButterworthHighpassFilterWidget, self).set_data_processor_input_field_value()

        self.cutoffLineEdit.setText(str(self.data_processor.cutoff))
        self.fsLineEdit.setText(str(self.data_processor.fs))
        self.orderLineEdit.setText(str(self.data_processor.order))

    def set_data_processor_input_field_constrain(self):
        self.cutoffLineEdit.setValidator(QDoubleValidator())
        self.fsLineEdit.setValidator(QDoubleValidator())
        self.orderLineEdit.setValidator(NoCommaIntValidator())

    def connect_data_processor_input_field_signal(self):
        super(ButterworthHighpassFilterWidget, self).connect_data_processor_input_field_signal()
        self.cutoffLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.orderLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        cutoff = self.get_cutoff()
        fs = self.get_fs()
        order = self.get_order()

        self.data_processor.set_data_processor_params(cutoff=cutoff, fs=fs, order=order)

    def get_cutoff(self):
        try:
            cutoff = abs(float(self.cutoffLineEdit.text()))
        except ValueError:
            return 0
        return cutoff

    def get_fs(self):
        try:
            fs = abs(float(self.fsLineEdit.text()))
        except ValueError:
            return 0
        return fs

    def get_order(self):
        try:
            order = abs(int(self.orderLineEdit.text()))
        except ValueError:
            return 0
        return order


class RootMeanSquareWidget(DataProcessorWidget):
    def __init__(self, parent, data_processor=None, adding_data_processor=False):
        if data_processor is None:
            data_processor = RootMeanSquare()
            data_processor.fs = float(get_stream_nominal_sampling_rate(parent.stream_name))


        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_RootMeanSquareWidget, self)

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(RootMeanSquareWidget, self).set_data_processor_input_field_value()

        self.fsLineEdit.setText(str(self.data_processor.fs))
        self.windowLineEdit.setText(str(self.data_processor.window))

    def set_data_processor_input_field_constrain(self):
        self.fsLineEdit.setValidator(QDoubleValidator())
        self.windowLineEdit.setValidator(QDoubleValidator())

    def connect_data_processor_input_field_signal(self):
        super(RootMeanSquareWidget, self).connect_data_processor_input_field_signal()
        self.fsLineEdit.textChanged.connect(self.data_processor_settings_on_changed)
        self.windowLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        fs = self.get_fs()
        window = self.get_window()

        self.data_processor.set_data_processor_params(fs=fs, window=window)

    def get_fs(self):
        try:
            fs = abs(float(self.fsLineEdit.text()))
        except ValueError:
            return 0
        return fs

    def get_window(self):
        try:
            window = abs(float(self.windowLineEdit.text()))
        except ValueError:
            return 0
        return window


class ClutterRemovalWidget(DataProcessorWidget):
    def __init__(self, parent, data_processor=None, adding_data_processor=False):
        if data_processor is None:
            data_processor = ClutterRemoval()

        super().__init__(parent, data_processor, adding_data_processor)
        self.ui = uic.loadUi(AppConfigs()._ui_ClutterRemovalWidget, self)

        ####################
        self.__post_init__()

    def set_data_processor_input_field_value(self):
        super(ClutterRemovalWidget, self).set_data_processor_input_field_value()

        self.signalClutterRatioLineEdit.setText(str(self.data_processor.signal_clutter_ratio))

    def set_data_processor_input_field_constrain(self):
        self.signalClutterRatioLineEdit.setValidator(QDoubleValidator())

    def connect_data_processor_input_field_signal(self):
        super(ClutterRemovalWidget, self).connect_data_processor_input_field_signal()
        self.signalClutterRatioLineEdit.textChanged.connect(self.data_processor_settings_on_changed)

    def set_data_processor_params(self):
        signal_clutter_ratio = self.get_signal_clutter_ratio()

        self.data_processor.set_data_processor_params(signal_clutter_ratio=signal_clutter_ratio)

    def get_signal_clutter_ratio(self):
        try:
            signal_clutter_ratio = float(self.signalClutterRatioLineEdit.text())
        except ValueError:
            return 0
        return signal_clutter_ratio


class DataProcessorWidgetType(Enum):
    NotchFilter = NotchFilterWidget
    ButterworthLowpassFilter = ButterworthLowpassFilterWidget
    ButterworthHighpassFilter = ButterworthHighpassFilterWidget
    ButterworthBandpassFilter = ButterworthBandPassFilterWidget
    RootMeanSquare = RootMeanSquareWidget
    ClutterRemoval = ClutterRemovalWidget
