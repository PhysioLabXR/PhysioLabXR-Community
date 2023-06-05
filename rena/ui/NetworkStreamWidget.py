# # This Python file uses the following encoding: utf-8
# import time
# from collections import deque
#
# import numpy as np
# from PyQt5 import QtWidgets, uic, QtCore
# from PyQt5.QtCore import QTimer, QThread, QMutex, Qt
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtWidgets import QDialogButtonBox, QSplitter
#
# from exceptions.exceptions import ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
# from rena import config, config_ui
# from rena.configs.configs import AppConfigs, LinechartVizMode
# from rena.presets.load_user_preset import create_default_group_entry
# from rena.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, get_stream_group_info, \
#     get_is_group_shown, pop_group_from_stream_preset, add_group_entry_to_stream, change_stream_group_order, \
#     change_stream_group_name, pop_stream_preset_from_settings, change_group_channels
# from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
# from rena.threadings import workers
# from rena.ui.GroupPlotWidget import GroupPlotWidget
# from rena.ui.PoppableWidget import Poppable
# from rena.ui.StreamOptionsWindow import StreamOptionsWindow
# from rena.ui.StreamWidget import StreamWidget
# from rena.ui.VizComponents import VizComponents
# from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
#     options_icon
# from rena.utils.buffers import DataBufferSingleStream
# from rena.utils.performance_utils import timeit
# from rena.utils.ui_utils import dialog_popup, clear_widget
#
#
# class NetworkStreamWidget(StreamWidget):
#
#     def __init__(self, parent_widget, parent_layout, stream_name, worker):
#         super().__init__(parent_widget, parent_layout, stream_name, worker)
#
#
#     def start_stop_stream_btn_clicked(self):
#         # check if is streaming
#         if self.worker.is_streaming:
#             self.worker.stop_stream()
#             if not self.worker.is_streaming:
#                 self.update_stream_availability(self.worker.is_stream_available)
#         else:
#             # self.reset_performance_measures()
#             try:
#                 self.worker.start_stream()
#                 self.worker.timestamp_queue.clear()
#             except LSLStreamNotFoundError as e:
#                 self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
#                 return
#             except ChannelMismatchError as e:  # only LSL's channel mismatch can be checked at this time, zmq's channel mismatch can only be checked when receiving data
#                 # self.main_parent.current_dialog = reply = QMessageBox.question(self, 'Channel Mismatch',
#                 #                              'The stream with name {0} found on the network has {1}.\n'
#                 #                              'The preset has {2} channels. \n '
#                 #                              'Do you want to reset your preset to a default and start stream.\n'
#                 #                              'You can edit your stream channels in Options if you choose No'.format(
#                 #                                  self.stream_name, e.message,
#                 #                                  len(get_stream_preset_info(self.stream_name, 'ChannelNames'))),
#                 #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
#                 preset_chan_num = len(get_stream_preset_info(self.stream_name, 'channel_names'))
#                 message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
#                 reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent, buttons=self.channel_mismatch_buttons)
#
#                 if reply.result():
#                     self.reset_preset_by_num_channels(e.message)
#                     try:
#                         self.worker.start_stream()  # start the stream again with updated preset
#                     except LSLStreamNotFoundError as e:
#                         self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
#                         return
#                 else:
#                     return
#             except Exception as e:
#                 raise UnsupportedErrorTypeError(str(e))
#             # if self.worker.is_streaming:
#             #     self.StartStopStreamBtn.setText("Stop Stream")
#         self.set_button_icons()
#         self.main_parent.update_active_streams()
#
#     def reset_channels
