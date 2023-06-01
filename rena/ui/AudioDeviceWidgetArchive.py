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
# from rena.presets.load_user_preset import create_default_group_entry
# from rena.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, get_stream_group_info, \
#     get_is_group_shown, pop_group_from_stream_preset, add_group_entry_to_stream, change_stream_group_order, \
#     change_stream_group_name, pop_stream_preset_from_settings, change_group_channels
# from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
# from rena.threadings import workers
# from rena.ui.GroupPlotWidget import GroupPlotWidget
# from rena.ui.PoppableWidget import Poppable
# from rena.ui.StreamOptionsWindow import StreamOptionsWindow
# from rena.ui.VizComponents import VizComponents
# from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
#     options_icon
# from rena.utils.buffers import DataBufferSingleStream
# from rena.utils.performance_utils import timeit
# from rena.utils.ui_utils import dialog_popup, clear_widget
#
#
# class AudioDeviceWidget(Poppable, QtWidgets.QWidget):
#     plot_format_changed_signal = QtCore.pyqtSignal(dict)
#     channel_mismatch_buttons = buttons=QDialogButtonBox.Yes | QDialogButtonBox.No
#
#     def __init__(self, parent_widget, parent_layout, stream_name, data_type, worker, networking_interface, port_number,
#                  insert_position=None, ):
#         """
#         StreamWidget is the main interface with plots and a single stream of data.
#         The stream can be either LSL or ZMQ.
#         @param parent_widget: the MainWindow
#         @param parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
#         """
#
#         # GUI elements
#         super().__init__(stream_name, parent_widget, parent_layout, self.remove_stream)
#         self.ui = uic.loadUi("ui/StreamContainer.ui", self)
#         self.set_pop_button(self.PopWindowBtn)
#
#         if type(insert_position) == int:
#             parent_layout.insertWidget(insert_position, self)
#         else:
#             parent_layout.addWidget(self)
#         self.parent = parent_layout
#         self.main_parent = parent_widget
#
#         ##
#         self.stream_name = stream_name  # this also keeps the subtopic name if using ZMQ
#         self.networking_interface = networking_interface
#         self.port_number = port_number
#         self.data_type = data_type
#         # self.preset = get_complete_stream_preset_info(self.stream_name)
#
#         self.actualSamplingRate = 0
#
#         self.StreamNameLabel.setText(stream_name)
#         self.OptionsBtn.setIcon(options_icon)
#         self.RemoveStreamBtn.setIcon(remove_stream_icon)
#
#         self.is_stream_available = False
#         self.in_error_state = False  # an error state to prevent ticking when is set to true
#
#         # visualization data buffer
#         self.current_timestamp = 0
#
#         # timer
#         self.timer = QTimer()
#         self.timer.setInterval(config.settings.value('pull_data_interval'))
#         self.timer.timeout.connect(self.ticks)
#
#         # visualization timer
#         self.v_timer = QTimer()
#         self.v_timer.setInterval(int(float(config.settings.value('visualization_refresh_interval'))))
#         self.v_timer.timeout.connect(self.visualize)
#
#         # connect btn
#         self.StartStopStreamBtn.clicked.connect(self.start_stop_stream_btn_clicked)
#         self.OptionsBtn.clicked.connect(self.options_btn_clicked)
#         self.RemoveStreamBtn.clicked.connect(self.remove_stream)
#
#         # inefficient loading of assets TODO need to confirm creating Pixmap in ui_shared result in crash
#         self.stream_unavailable_pixmap = QPixmap('../media/icons/streamwidget_stream_unavailable.png')
#         self.stream_available_pixmap = QPixmap('../media/icons/streamwidget_stream_available.png')
#         self.stream_active_pixmap = QPixmap('../media/icons/streamwidget_stream_viz_active.png')
#
#         # visualization component
#         # This variable stores all the visualization components we initialize it in the init_stream_visualization()
#         self.viz_components = None
#         self.num_points_to_plot = None
#
#         # data elements
#         self.viz_data_buffer = None
#         self.create_buffer()
#
#         # if (stream_srate := interface.get_nominal_srate()) != nominal_sampling_rate and stream_srate != 0:
#         #     print('The stream {0} found in LAN has sampling rate of {1}, '
#         #           'overriding in settings: {2}'.format(lsl_name, stream_srate, nominal_sampling_rate))
#         #     config.settings.setValue('NominalSamplingRate', stream_srate)
#
#         # load default settings from settings
#
#         self.worker_thread = QThread(self)
#         if self.networking_interface == 'LSL':
#             channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
#             self.worker = workers.LSLInletWorker(self.stream_name, channel_names, data_type=data_type, RenaTCPInterface=None)
#         elif self.networking_interface == 'ZMQ':
#             self.worker = workers.ZMQWorker(port_number=port_number, subtopic=stream_name, data_type=data_type)
#         elif self.networking_interface == 'Device':
#             assert worker
#             self.worker = worker
#         self.worker.signal_data.connect(self.process_stream_data)
#         self.worker.signal_stream_availability.connect(self.update_stream_availability)
#         self.worker.moveToThread(self.worker_thread)
#         self.worker_thread.start()
#
#         # create option window
#         self.stream_options_window = StreamOptionsWindow(parent_stream_widget=self, stream_name=self.stream_name, plot_format_changed_signal=self.plot_format_changed_signal)
#         self.stream_options_window.bar_chart_range_on_change_signal.connect(self.bar_chart_range_on_change)
#         self.stream_options_window.hide()
#
#         # create visualization component, must be after the option window ##################
#         self.channel_index_plot_widget_dict = {}
#         self.group_name_plot_widget_dict = {}
#         # add splitter to the layout
#         self.splitter = QSplitter(Qt.Vertical)
#         self.viz_group_scroll_layout.addWidget(self.splitter)
#
#         self.create_visualization_component()
#
#         self._has_new_viz_data = False
#
#         # FPS counter``
#         self.tick_times = deque(maxlen=10 * int(float(config.settings.value('visualization_refresh_interval'))))
#
#         # mutex for not update the settings while plotting
#         self.setting_update_viz_mutex = QMutex()
#
#         self.set_button_icons()
#         # start the timers
#         self.timer.start()
#         self.v_timer.start()
#
#         # Attributes purely for performance checks x############################
#         """
#         These attributes should be kept only on this perforamnce branch
#         """
#         self.update_buffer_times = []
#         self.plot_data_times = []
#         ########################################################################
#
#     def reset_performance_measures(self):
#         self.update_buffer_times = []
#         self.plot_data_times = []
#
#     def update_stream_availability(self, is_stream_available):
#         '''
#         this function check if the stream is available
#         '''
#         print('Stream {0} availability is {1}'.format(self.stream_name, is_stream_available), end='\r')
#         self.is_stream_available = is_stream_available
#         if self.worker.is_streaming:
#             if is_stream_available:
#                 if not self.StartStopStreamBtn.isEnabled(): self.StartStopStreamBtn.setEnabled(True)
#                 self.StreamAvailablilityLabel.setPixmap(self.stream_active_pixmap)
#                 self.StreamAvailablilityLabel.setToolTip("Stream {0} is being plotted".format(self.stream_name))
#             else:
#                 self.start_stop_stream_btn_clicked()  # must stop the stream before dialog popup
#                 self.set_stream_unavailable()
#                 self.main_parent.current_dialog = dialog_popup('Lost connection to {0}'.format(self.stream_name), title='Warning', mode='modeless')
#         else:
#             # is the stream is not available
#             if is_stream_available:
#                 self.set_stream_available()
#             else:
#                 self.set_stream_unavailable()
#         self.main_parent.update_active_streams()
#
#     def set_stream_unavailable(self):
#         self.StartStopStreamBtn.setEnabled(False)
#         self.StreamAvailablilityLabel.setPixmap(self.stream_unavailable_pixmap)
#         self.StreamAvailablilityLabel.setToolTip("Stream {0} is not available".format(self.stream_name))
#
#     def set_stream_available(self):
#         self.StartStopStreamBtn.setEnabled(True)
#         self.StreamAvailablilityLabel.setPixmap(self.stream_available_pixmap)
#         self.StreamAvailablilityLabel.setToolTip("Stream {0} is available to start".format(self.stream_name))
#
#     def set_button_icons(self):
#         if not self.is_streaming():
#             self.StartStopStreamBtn.setIcon(start_stream_icon)
#         else:
#             self.StartStopStreamBtn.setIcon(stop_stream_icon)
#
#         if not self.is_popped:
#             self.PopWindowBtn.setIcon(pop_window_icon)
#         else:
#             self.PopWindowBtn.setIcon(dock_window_icon)
#
#     def options_btn_clicked(self):
#         print("Option window button clicked")
#         self.stream_options_window.show()
#         self.stream_options_window.activateWindow()
#
#     def group_plot_widget_edit_option_clicked(self, group_name: str):
#         self.options_btn_clicked()
#         self.stream_options_window.set_selected_group(group_name)
#
#     def is_streaming(self):
#         return self.worker.is_streaming
#
#     def start_stop_stream_btn_clicked(self):
#         # check if is streaming
#         if self.worker.is_streaming:
#             self.worker.stop_stream()
#             if not self.worker.is_streaming:
#                 # started
#                 # print("sensor stopped")
#                 # self.StartStopStreamBtn.setText("Start Stream")  # toggle the icon
#                 self.update_stream_availability(self.worker.is_stream_available)
#         else:
#             # self.reset_performance_measures()
#             try:
#                 self.worker.start_stream()
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
#     def reset_preset_by_num_channels(self, num_channels):
#         pop_stream_preset_from_settings(self.stream_name)
#         self.main_parent.create_preset(self.stream_name, self.data_type, self.port_number, self.networking_interface, num_channels=num_channels)  # update preset in settings
#         self.create_buffer()  # recreate the interface and buffer, using the new preset
#         self.worker.reset_interface(self.stream_name, get_stream_preset_info(self.stream_name, 'channel_names'))
#
#         self.stream_options_window.reload_preset_to_UI()
#         self.reset_viz()
#
#     def reset_viz(self):
#         """
#         caller to this function must ensure self.group_info is modified and up to date with user changes
#         """
#         self.clear_stream_visualizations()
#         self.create_visualization_component()
#
#     def create_buffer(self):
#         channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
#         buffer_size = 1 if len(channel_names) > config.MAX_TIMESERIES_NUM_CHANNELS_PER_STREAM else config.VIZ_DATA_BUFFER_MAX_SIZE
#         self.viz_data_buffer = DataBufferSingleStream(num_channels=len(channel_names),
#                                                       buffer_sizes=buffer_size, append_zeros=True)
#
#     def remove_stream(self):
#
#         if self.main_parent.recording_tab.is_recording:
#             self.main_parent.current_dialog = dialog_popup(msg='Cannot remove stream while recording.')
#             return False
#         self.timer.stop()
#         self.v_timer.stop()
#         if self.worker.is_streaming:
#             self.worker.stop_stream()
#         self.worker_thread.exit()
#         self.worker_thread.wait()  # wait for the thread to exit
#
#         self.main_parent.stream_widgets.pop(self.stream_name)
#         self.main_parent.remove_stream_widget(self)
#         # close window if popped
#         if self.is_popped:
#             self.delete_window()
#         self.deleteLater()
#         self.stream_options_window.close()
#         return True
#
#     def update_channel_shown(self, channel_index, is_shown, group_name):
#         channel_plot_widget = self.channel_index_plot_widget_dict[channel_index]
#         channel_plot_widget.show() if is_shown else channel_plot_widget.hide()
#         self.update_groups_shown(group_name)
#
#     def update_groups_shown(self, group_name):
#         # assuming group info is update to date with in the persist settings
#         # check if there's active channels in this group
#         if get_is_group_shown(self.stream_name, group_name):
#             self.group_name_plot_widget_dict[group_name].show()
#         else:
#             self.group_name_plot_widget_dict[group_name].hide()
#
#     def clear_stream_visualizations(self):
#         self.channel_index_plot_widget_dict = {}
#         self.group_name_plot_widget_dict = {}
#         clear_widget(self.splitter)
#
#     def init_stream_visualization(self):
#         channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
#
#         group_plot_widget_dict = {}
#         group_info = get_stream_group_info(self.stream_name)
#         for group_name in group_info.keys():
#             # if group_info[group_name].is_only_image_enabled:
#             #     update_selected_plot_format(self.stream_name, group_name, 1)  # change the plot format to image now
#             #     group_info = get_stream_group_info(self.stream_name)  # reload the group info from settings
#
#             group_channel_names = [channel_names[int(i)] for i in group_info[group_name].channel_indices]
#             group_plot_widget_dict[group_name] = GroupPlotWidget(self, self.stream_name, group_name, group_channel_names, get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'), self.plot_format_changed_signal)
#             self.splitter.addWidget(group_plot_widget_dict[group_name])
#             self.num_points_to_plot = self.get_num_points_to_plot()
#
#         return group_plot_widget_dict
#
#     def create_visualization_component(self):
#         group_plot_dict = self.init_stream_visualization()
#         self.viz_components = VizComponents(self.fs_label, self.ts_label, group_plot_dict)
#
#     def process_stream_data(self, data_dict):
#         '''
#         update the visualization buffer, recording buffer, and scripting buffer
#         '''
#         if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
#             try:
#                 self.update_buffer_times.append(timeit(self.viz_data_buffer.update_buffer, (data_dict, ))[1])  # NOTE performance test scripts, don't include in production code
#                 self._has_new_viz_data = True
#                 # self.viz_data_buffer.update_buffer(data_dict)
#             except ChannelMismatchError as e:
#                 self.in_error_state = True
#                 preset_chan_num = len(get_stream_preset_info(self.stream_name, 'channel_names'))
#                 message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
#                 reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent, buttons=self.channel_mismatch_buttons)
#
#                 if reply.result():
#                     self.reset_preset_by_num_channels(e.message)
#                     self.in_error_state = False
#                     return
#                 else:
#                     self.start_stop_stream_btn_clicked()  # stop the stream
#                     self.in_error_state = False
#                     return
#             self.actualSamplingRate = data_dict['sampling_rate']
#             self.current_timestamp = data_dict['timestamps'][-1]
#             # notify the internal buffer in recordings tab
#
#             # reshape data_dict based on sensor interface
#             self.main_parent.recording_tab.update_recording_buffer(data_dict)
#             self.main_parent.scripting_tab.forward_data(data_dict)
#             # scripting tab
#
#     '''
#     settings on change:
#     visualization can be changed while recording with mutex
#     1. lock settings on change
#     2. update visualization
#     3. save changes to RENA_Settings
#     4. release mutex
#
#     data processing cannot be changed while recording
#
#     # cannot add channels while streaming/recording
#
#
#     '''
#
#     def stream_settings_changed(self, change):
#         self.setting_update_viz_mutex.lock()
#         # resolve the
#         if change[0] == "nominal_sampling_rate":
#             pass  # TODO
#         # TODO add other changes such as plot format, plot order, etc...
#
#         self.setting_update_viz_mutex.unlock()
#
#     def visualize(self):
#         '''
#         This is the function for LSL data visualization.
#         It plot the data from the data visualization buffer based on the configuration
#         The data to plot is in the parameter self.viz_data_buffer
#         '''
#
#         self.tick_times.append(time.time())
#         # print("Viz FPS {0}".format(self.get_fps()), end='\r')
#         self.worker.signal_stream_availability_tick.emit()  # signal updating the stream availability
#         # for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
#         actual_sampling_rate = self.actualSamplingRate
#         # max_display_datapoint_num = self.stream_widget_visualization_component.plot_widgets[0].size().width()
#
#         # reduce the number of points to plot to the number of pixels in the corresponding plot widget
#
#         # if data_to_plot.shape[-1] > config.DOWNSAMPLE_MULTIPLY_THRESHOLD * max_display_datapoint_num:
#         #     data_to_plot = np.nan_to_num(data_to_plot, nan=0)
#         #     # start = time.time()
#         #     # data_to_plot = data_to_plot[:, ::int(data_to_plot.shape[-1] / max_display_datapoint_num)]
#         #     # data_to_plot = signal.resample(data_to_plot, int(data_to_plot.shape[-1] / max_display_datapoint_num), axis=1)
#         #     data_to_plot = decimate(data_to_plot, q=int(data_to_plot.shape[-1] / max_display_datapoint_num),
#         #                             axis=1)  # resample to 100 hz with retain history of 10 sec
#         #     # print(time.time()-start)
#         #     time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, num=data_to_plot.shape[-1])
#
#         # self.LSL_plots_fs_label_dict[lsl_stream_name][2].setText(
#         #     'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
#         #
#         # [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
#         #  enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]
#         # change to loop with type condition plot time_series and image
#         # if self.LSL_plots_fs_label_dict[lsl_stream_name][3]:
#         # plot_channel_num_offset = 0
#         if not self._has_new_viz_data:
#             return
#         self.viz_data_buffer.buffer[0][np.isnan(self.viz_data_buffer.buffer[0])] = 0  # zero out nan
#         data_to_plot = self.viz_data_buffer.buffer[0][:, -self.num_points_to_plot:]
#
#         for plot_group_index, (group_name) in enumerate(get_stream_group_info(self.stream_name).keys()):
#             # self.plot_data_times.append(timeit(self.viz_components.group_plots[group_name].plot_data, (data_to_plot, ))[1])  # NOTE performance test scripts, don't include in production code
#             self.viz_components.group_plots[group_name].plot_data(data_to_plot)
#
#         # show the label
#         self.viz_components.fs_label.setText(
#             'Sampling rate = {:.3f}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
#         self.viz_components.ts_label.setText('Current Time Stamp = {:.3f}'.format(self.current_timestamp))
#
#         self._has_new_viz_data = False
#
#     def ticks(self):
#         self.worker.signal_data_tick.emit()
#         #     self.recent_tick_refresh_timestamps.append(time.time())
#         #     if len(self.recent_tick_refresh_timestamps) > 2:
#         #         self.tick_rate = 1 / ((self.recent_tick_refresh_timestamps[-1] - self.recent_tick_refresh_timestamps[0]) / (
#         #                     len(self.recent_tick_refresh_timestamps) - 1))
#         #
#         #     self.tickFrequencyLabel.setText(
#         #         'Pull Data Frequency: {0}'.format(round(self.tick_rate, config_ui.tick_frequency_decimal_places)))
#
#     def init_server_client(self):
#         print('John')
#
#         # dummy preset for now:
#         stream_name = 'OpenBCI'
#         port_id = int(time.time())
#         identity = 'server'
#         processor_dic = {}
#         rena_tcp_request_object = RenaTCPAddDSPWorkerRequestObject(stream_name, port_id, identity, processor_dic)
#         self.main_parent.rena_dsp_client.send_obj(rena_tcp_request_object)
#         rena_tcp_request_object = self.main_parent.rena_dsp_client.recv_obj()
#         print('DSP worker created')
#         self.dsp_client_interface = RenaTCPInterface(stream_name=stream_name, port_id=port_id, identity='client')
#
#         # send to server
#
#     def get_fps(self):
#         try:
#             return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
#         except (ZeroDivisionError, IndexError) as e:
#             return 0
#
#     def is_widget_streaming(self):
#         return self.worker.is_streaming
#
#     def on_num_points_to_display_change(self):
#         '''
#         this function is called by StreamOptionWindow when user change number of points to plot by changing
#         the sampling rate or the display duration.
#         Changing the num points to plot here will cause, in the next plot cycle, GroupPlotWidget will have a mismatch between
#         the data received from here and its time vector. This will cause the GroupPlotWidget to update its time vector
#         :param new_sampling_rate:
#         :param new_display_duration:
#         :return:
#         '''
#         self.num_points_to_plot = self.get_num_points_to_plot()
#         if self.viz_components is not None:
#             self.viz_components.update_nominal_sampling_rate()
#
#     # def reload_visualization_elements(self, info_dict):
#     #     self.group_info = collect_stream_all_groups_info(self.stream_name)
#     #     clear_layout(self.MetaInfoVerticalLayout)
#     #     clear_layout(self.TimeSeriesPlotsLayout)
#     #     clear_layout(self.ImageWidgetLayout)
#     #     clear_layout(self.BarPlotWidgetLayout)
#     #     self.create_visualization_component()
#
#     # @QtCore.pyqtSlot(dict)
#     # def plot_format_on_change(self, info_dict):
#         # old_format = self.group_info[info_dict['group_name']]['selected_plot_format']
#         # self.group_info[info_dict['group_name']]['selected_plot_format'] = info_dict['new_format']
#
#         # self.preset_on_change()  # update the group info
#
#         # self.viz_components.group_plots[plot_format_index_dict[old_format]][
#         #     info_dict['group_name']].hide()
#         # self.viz_components.group_plots[plot_format_index_dict[info_dict['new_format']]][
#         #     info_dict['group_name']].show()
#
#         # update the plot hide display
#
#     # @QtCore.pyqtSlot(dict)
#     # def image_changed(self, change: dict):
#     #     if change['group_name'] in self.group_name_plot_widget_dict.keys():
#     #         self.group_name_plot_widget_dict[change['group_name']].update_image_info(change['this_group_info_image'])
#
#     # def preset_on_change(self):
#     #     self.group_info = get_stream_group_info(self.stream_name)  # reload the group info
#
#     # def get_image_format_and_shape(self, group_name):
#     #     width = self.group_info[group_name]['plot_format']['image']['width']
#     #     height = self.group_info[group_name]['plot_format']['image']['height']
#     #     image_format = self.group_info[group_name]['plot_format']['image']['image_format']
#     #     depth = image_depth_dict[image_format]
#     #     channel_format = self.group_info[group_name]['plot_format']['image']['channel_format']
#     #     scaling_factor = self.group_info[group_name]['plot_format']['image']['scaling_factor']
#     #
#     #     return width, height, depth, image_format, channel_format, scaling_factor
#
#     #############################################
#
#     def bar_chart_range_on_change(self, group_name):
#         self.viz_components.group_plots[group_name].update_bar_chart_range()
#
#     #############################################
#
#     def channel_group_changed(self, change_dict):
#         """
#         Called when one or more channel's parent group is changed
#         @param change_dict:
#         """
#         # update the group info
#         for group_name, child_channels in change_dict.items():
#             if len(child_channels) == 0:
#                 pop_group_from_stream_preset(self.stream_name, group_name)
#             else:  # cover the cases for both changed groups and new group
#                 channel_indices = [x.lsl_index for x in child_channels]
#                 is_channels_shown = [x.is_shown for x in child_channels]
#                 if group_name not in get_stream_group_info(self.stream_name).keys():  # if this is a new group
#                     add_group_entry_to_stream(self.stream_name, create_default_group_entry(len(child_channels), group_name, channel_indices=channel_indices, is_channels_shown=is_channels_shown))
#                 else:
#                     change_group_channels(self.stream_name, group_name, channel_indices, is_channels_shown)
#         # save_preset()
#         self.reset_viz()
#
#     def group_order_changed(self, group_order):
#         """
#         Called when the group order is changed
#         @param group_order:
#         """
#         change_stream_group_order(self.stream_name, group_order)
#         # save_preset()
#         self.reset_viz()
#
#     def change_group_name(self, new_group_name, old_group_name):
#         try:
#             change_stream_group_name(self.stream_name, new_group_name, old_group_name)
#         except ValueError as e:
#             dialog_popup(str(e), mode='modeless')
#         self.viz_components.group_plots[new_group_name] = self.viz_components.group_plots.pop(old_group_name)
#         self.viz_components.group_plots[new_group_name].change_group_name(new_group_name)
#
#     def change_channel_name(self, group_name, new_ch_name, old_ch_name, lsl_index):
#         # change channel name in the settings
#         channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
#         changing_channel_index = channel_names.index(old_ch_name)
#         channel_names[changing_channel_index] = new_ch_name
#         set_stream_preset_info(self.stream_name, 'channel_names', channel_names)
#
#         # change the name in the plots
#         self.viz_components.group_plots[group_name].change_channel_name(new_ch_name, old_ch_name, lsl_index)
#
#     def get_num_points_to_plot(self):
#         display_duration = get_stream_preset_info(self.stream_name, 'display_duration')
#         return int(display_duration * get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'))
#
#     def get_pull_data_delay(self):
#         return self.worker.get_pull_data_delay()
#
#     def set_spectrogram_cmap(self, group_name):
#         self.viz_components.set_spectrogram_cmap(group_name)
#
#     def try_close(self):
#         return self.remove_stream()


import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer

import rena.threadings.ScreenCaptureWorker
import rena.threadings.WebcamWorker
from rena import config
from rena.config import settings
from rena.presets.presets_utils import get_video_scale, get_video_channel_order, is_video_webcam, get_video_device_id, \
    get_audio_device_index
from rena.threadings.AudioDeviceWorker import AudioDeviceWorker
from rena.ui.PoppableWidget import Poppable
from rena.ui.VideoDeviceOptions import VideoDeviceOptions
from rena.ui_shared import remove_stream_icon, \
    options_icon
from rena.utils.ui_utils import dialog_popup


class AudioDeviceWidgetArchive(Poppable, QtWidgets.QWidget):
    def __init__(self, parent_widget, parent_layout, audio_device_name, insert_position=None):
        """

        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(audio_device_name, parent_widget, parent_layout, self.remove_audio_device)

        self.audio_device_name = audio_device_name
        self.audio_device_index = get_audio_device_index(audio_device_name)

        self.ui = uic.loadUi("ui/AudioDeviceWidgetArchive.ui", self)
        self.set_pop_button(self.PopWindowBtn)

        if type(insert_position) == int:
            parent_layout.insertWidget(insert_position, self)
        else:
            parent_layout.addWidget(self)
        self.parent_layout = parent_layout
        self.main_parent = parent_widget

        self.AudioDeviceNameLabel.setText(self.audio_device_name)

        self.RemoveAudioBtn.clicked.connect(self.remove_audio_device)
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveAudioBtn.setIcon(remove_stream_icon)

        self.tick_times = deque(maxlen=10 * int(float(config.settings.value('visualization_refresh_interval'))))

        # TODO: audio visualization
        self.worker_thread = pg.QtCore.QThread(self)
        self.worker = AudioDeviceWorker(self.audio_device_index)
        self.worker.start_stream()
        self.worker.signal_data.connect(self.process_stream_data)

        # connect button
        # self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        # self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        #####################################################
        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.settings.value('pull_data_interval'))
        self.timer.timeout.connect(self.ticks)

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(int(float(config.settings.value('visualization_refresh_interval'))))
        self.v_timer.timeout.connect(self.visualize)



        self.worker_thread.start()
        self.timer.start()
        self.v_timer.start()


    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0





    def init_audio_stream_widget(self):

        pass


    def process_stream_data(self, data_dict):
        pass

    def visualize(self):
        pass

    def start_audio_stream(self):
        self.worker.start_stream()

    def stop_audio_stream(self):
        self.worker.stop_stream()

    def remove_audio_device(self):
        self.stop_audio_stream()

    def ticks(self):
        self.worker.signal_data_tick.emit()

    def try_close(self):
        return self.remove_audio_device()




    #     self.ui = uic.loadUi("ui/VideoDeviceWidget.ui", self)
    #     self.set_pop_button(self.PopWindowBtn)
    #
    #     if type(insert_position) == int:
    #         parent_layout.insertWidget(insert_position, self)
    #     else:
    #         parent_layout.addWidget(self)
    #     self.parent_layout = parent_layout
    #     self.main_parent = parent_widget
    #     self.video_device_name = video_device_name
    #     self.VideoDeviceNameLabel.setText(self.video_device_name)
    #
    #     # check if the video device is a camera or screen capture ####################################
    #     self.is_webcam = is_video_webcam(self.video_device_name)
    #     self.video_device_long_name = ('Webcam ' if self.is_webcam else 'Screen Capture ') + str(video_device_name)
    #     # Connect UIs ##########################################
    #     self.RemoveVideoBtn.clicked.connect(self.remove_video_device)
    #     self.OptionsBtn.setIcon(options_icon)
    #     self.RemoveVideoBtn.setIcon(remove_stream_icon)
    #
    #     # FPS counter``
    #     self.tick_times = deque(maxlen=10 * settings.value('video_device_refresh_interval'))
    #
    #     # image label
    #     self.plot_widget = pg.PlotWidget()
    #     self.ImageWidget.layout().addWidget(self.plot_widget)
    #     self.plot_widget.enableAutoRange(enable=False)
    #     self.image_item = pg.ImageItem()
    #     self.plot_widget.addItem(self.image_item)
    #
    #     # create options window ##########################################
    #     self.video_options_window = VideoDeviceOptions(parent_stream_widget=self, video_device_name=self.video_device_name)
    #     self.video_options_window.hide()
    #     self.OptionsBtn.clicked.connect(lambda: (self.video_options_window.show(), self.video_options_window.activateWindow()))
    #
    #     # worker and worker threads ##########################################
    #     self.worker_thread = pg.QtCore.QThread(self)
    #
    #     video_scale, channel_order = get_video_scale(self.video_device_name), get_video_channel_order(self.video_device_name)
    #     if self.is_webcam:
    #         self.worker = rena.threadings.WebcamWorker.WebcamWorker(get_video_device_id(video_device_name), video_scale, channel_order)
    #     else:
    #         self.worker = rena.threadings.ScreenCaptureWorker.ScreenCaptureWorker(video_device_name, video_scale, channel_order)
    #     self.worker.change_pixmap_signal.connect(self.visualize)
    #     self.worker.moveToThread(self.worker_thread)
    #
    #     # define timer ##########################################
    #     self.timer = QTimer()
    #     self.timer.setInterval(settings.value('video_device_refresh_interval'))
    #     self.timer.timeout.connect(self.ticks)
    #
    #     self.worker_thread.start()
    #     self.timer.start()
    #
    #     self.is_image_fitted_to_frame = False
    #
    #
    # def visualize(self, cam_id_cv_img_timestamp):
    #     self.tick_times.append(time.time())
    #     cam_id, image, timestamp = cam_id_cv_img_timestamp
    #     # qt_img = convert_rgb_to_qt_image(image)
    #     image = np.swapaxes(image, 0, 1)
    #     self.image_item.setImage(image)
    #
    #     if not self.is_image_fitted_to_frame:
    #         self.plot_widget.setXRange(0, image.shape[0])
    #         self.plot_widget.setYRange(0, image.shape[1])
    #         self.is_image_fitted_to_frame = True
    #
    #     # self.ImageLabel.setPixmap(qt_img)
    #     self.main_parent.recording_tab.update_camera_screen_buffer(cam_id, image, timestamp)
    #
    # def remove_video_device(self):
    #     if self.main_parent.recording_tab.is_recording:
    #         dialog_popup(msg='Cannot remove stream while recording.')
    #         return False
    #     self.timer.stop()
    #     self.worker.stop_stream()
    #     self.worker_thread.exit()
    #     self.worker_thread.wait()  # wait for the thread to exit
    #
    #     self.main_parent.video_device_widgets.pop(self.video_device_name)
    #     self.main_parent.remove_stream_widget(self)
    #
    #     # close window if popped
    #     if self.is_popped:
    #         self.delete_window()
    #     self.deleteLater()
    #     return True
    #
    # def ticks(self):
    #     self.worker.tick_signal.emit()
    #
    # def get_fps(self):
    #     try:
    #         return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
    #     except (ZeroDivisionError, IndexError) as e:
    #         return 0
    #
    # def get_pull_data_delay(self):
    #     return self.worker.get_pull_data_delay()
    #
    # def is_widget_streaming(self):
    #     return self.worker.is_streaming
    #
    # def video_preset_changed(self):
    #     self.worker.video_scale = get_video_scale(self.video_device_name)
    #     self.worker.channel_order = get_video_channel_order(self.video_device_name)
    #     self.is_image_fitted_to_frame = False
    #
    # def try_close(self):
    #     return self.remove_video_device()
