# This Python file uses the following encoding: utf-8
import time

from PyQt5.QtCore import QTimer

from exceptions.exceptions import ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from rena import config_ui
from rena.configs.configs import AppConfigs, LinechartVizMode
from rena.presets.Presets import PresetType
from rena.presets.load_user_preset import create_default_group_entry
from rena.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, get_stream_group_info, \
    get_is_group_shown, pop_group_from_stream_preset, add_group_entry_to_stream, change_stream_group_order, \
    change_stream_group_name, pop_stream_preset_from_settings, change_group_channels
from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
from rena.threadings import workers
from rena.ui.BaseStreamWidget import BaseStreamWidget
from rena.ui.GroupPlotWidget import GroupPlotWidget
from rena.ui.VizComponents import VizComponents
from rena.utils.buffers import DataBufferSingleStream
from rena.utils.performance_utils import timeit
from rena.utils.ui_utils import dialog_popup, clear_widget


class ZMQWidget(BaseStreamWidget):

    def __init__(self, parent_widget, parent_layout, topic_name, port_number, data_type, insert_position=None):
        """
        BaseStreamWidget is the main interface with plots and a single stream of data.
        @param parent_widget: the MainWindow class
        @param parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
        @param stream_name: the name of the stream
        """

        # GUI elements
        super().__init__(parent_widget, parent_layout, PresetType.ZMQ, topic_name,
                         data_timer_interval=AppConfigs().pull_data_interval, use_viz_buffer=True, insert_position=insert_position)
        self.data_type = data_type
        self.port = port_number

        zmq_worker = workers.ZMQWorker(port_number=port_number, subtopic=topic_name, data_type=data_type)
        self.connect_worker(zmq_worker, True)
        self.connect_start_stop_btn(self.start_stop_stream_btn_clicked)
        self.start_timers()

    def start_stop_stream_btn_clicked(self):
        try:
            super().start_stop_stream_btn_clicked()
        except Exception as e:
            raise UnsupportedErrorTypeError(str(e))

    def process_stream_data(self, data_dict):
        try:
            super().process_stream_data(data_dict)
        except ChannelMismatchError as e:
            self.in_error_state = True
            preset_chan_num = len(get_stream_preset_info(self.stream_name, 'channel_names'))
            message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
            reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent,
                                 buttons=self.channel_mismatch_buttons)
            if reply.result():
                self.reset_preset_by_num_channels(e.message, self.data_type, port=self.port)
                self.in_error_state = False
                return
            else:
                self.StartStopStreamBtn.click()  # stop the stream
                self.in_error_state = False
                return
