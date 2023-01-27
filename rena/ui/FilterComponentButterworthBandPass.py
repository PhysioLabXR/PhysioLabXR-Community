# This Python file uses the following encoding: utf-8

# This Python file uses the following encoding: utf-8
import uuid

from PyQt5 import QtCore, QtWidgets
from PyQt5 import uic
from PyQt5.QtGui import QIntValidator, QDoubleValidator

from rena import config
from rena.shared import FilterType
from rena.utils.rena_dsp_utils import RealtimeButterBandpass, RenaFilter


# This Python file uses the following encoding: utf-8

# cannot remove group if there is active filters

# args: {
#  filter_name
#  filter_index
#  fs
#  lowcut
#  highcut
#  order
#  filter_valid
#  channel_num
#  // depends on the group name
# }


class FilterComponentButterworthBandPass(QtWidgets.QWidget):
    filter_on_change_signal = QtCore.pyqtSignal(RenaFilter)

    def __init__(self, stream_preset=None, args=None):
        super().__init__()
        self.ui = uic.loadUi("ui/FilterComponentButterworthBandPass.ui", self)
        self.rena_filter = RealtimeButterBandpass()
        self.filter_type = FilterType.ButterworthBandPass
        self.args = args

        # low cut
        if args!=None:
            self.id = args['id']
        else:
            self.id = uuid.uuid4()
            self.export_filter_args_to_settings()

        self.samplingFrequencyLineEdit.setValidator(QDoubleValidator())
        self.lowCutFrequencyLineEdit.setValidator(QDoubleValidator())
        self.highCutFrequencyLineEdit.setValidator(QDoubleValidator())
        self.filterOrderLineEdit.setValidator(QIntValidator())

        self.lowCutFrequencyLineEdit.textChanged.connect(self.filter_args_on_change)
        self.highCutFrequencyLineEdit.textChanged.connect(self.filter_args_on_change)
        self.filterOrderLineEdit.textChanged.connect(self.filter_args_on_change)

        self.removeFilterBtn.clicked.connect(self.remove_filter_button)

    def import_filter_args(self, args):
        pass

    def export_filter_args_to_settings(self, stream_name, group_name):
        # filter info
        config.settings.beginGroup('presets/streampresets/{0}/{1}/{2}/'.
                                   format(stream_name, group_name, 'filter_info'))
        # config.



    def get_filter_args(self):
        args = {}
        args['id'] = self.id
        args['filter_type'] = self.filter_type
        args['fs'] = self.get_filter
        args['lowcut'] = self.get_filter_low_cut_frequency()
        args['highcut'] = self.get_filter_high_cut_frequency()
        args['order'] = self.get_filter_order()

    def get_group_info(self):
        pass

    def filter_process_buffer(self, input_buffer):
        output = self.rena_filter.process_buffer(input_buffer)
        return output

    def remove_filter_button(self):
        self.deleteLater()

    def init_filter(self, args):
        self.rena_filter.__init__(lowcut=args['lowcut'],
                                  highcut=args['hightcut'],
                                  fs=args['fs'],
                                  order=args['order'],
                                  channel_num=args['channel_num'])


    def filter_args_on_change(self):
        self.args['fs'] = self.get_filter_sampling_frequency()
        self.args['highcut'] = self.get_filter_high_cut_frequency()
        self.args['lowcut'] = self.get_filter_low_cut_frequency()
        self.args['order'] = self.get_filter_order()

        self.filter_args_valid()
        # if filter valid, send to the worker
        if self.args['filter_valid']:
            self.filter_on_change_signal.emit(self.rena_filter)

    def get_filter_sampling_frequency(self):
        try:
            sampling_frequency = abs(float(self.samplingFrequencyLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return sampling_frequency


    def get_filter_high_cut_frequency(self):
        try:
            high_cut_frequency = abs(float(self.highCutFrequencyLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return high_cut_frequency

    def get_filter_low_cut_frequency(self):
        try:
            low_cut_frequency = abs(float(self.lowCutFrequencyLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return low_cut_frequency

    def get_filter_order(self):
        try:
            order = abs(float(self.filterOrderLineEdit.text()))
        except ValueError:  # in case the string cannot be convert to a float
            return 0
        return order

    def filter_args_valid(self):
        if self.low_cut_frequency > self.high_cut_frequency:
            self.filterInfoLabel.setText("low cut frequency > high cut frequency")
            self.args['filter_valid'] = False
        else:
            try:
                self.rena_filter = \
                    RealtimeButterBandpass(highcut=self.args['highcut'],
                                           lowcut=self.args['lowcut'], order=self.args['order'], channel_num=self.args['channel_num'])
                self.filterInfoLabel.setText("filter valid")
                self.args['filter_valid'] = True
            except:
                self.args['filter_valid'] = False
                self.filterInfoLabel.setText("Invalid filter design")
                print("invalid filter design")

        if self.args['filter_valid']:
            self.filterInfoLabel.setStyleSheet('color: green')
        else:
            self.filterInfoLabel.setStyleSheet('color: red')

