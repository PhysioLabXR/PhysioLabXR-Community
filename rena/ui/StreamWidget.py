# This Python file uses the following encoding: utf-8
import time
from collections import deque

from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QTimer, QThread, QMutex
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox

from exceptions.exceptions import ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from rena import config, config_ui
from rena.config_ui import image_depth_dict
from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
from rena.threadings import workers
from rena.ui.GroupPlotWidget import GroupPlotWidget
from rena.ui.StreamOptionsWindow import StreamOptionsWindow
from rena.ui.VizComponents import VizComponents
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
    options_icon
from rena.utils.buffers import DataBufferSingleStream
from rena.utils.performance_utils import timeit
from rena.utils.settings_utils import get_stream_preset_info, \
    collect_stream_all_groups_info, is_group_shown, remove_stream_preset_from_settings, \
    set_stream_preset_info, update_selected_plot_format, export_group_info_to_settings, create_default_group_info, \
    change_group_name
from rena.utils.ui_utils import AnotherWindow, dialog_popup, clear_layout


class StreamWidget(QtWidgets.QWidget):
    plot_format_changed_signal = QtCore.pyqtSignal(dict)
    def __init__(self, main_parent, parent, stream_name, data_type, worker, networking_interface, port_number,
                 insert_position=None, ):
        """
        LSL interface is created in StreamWidget
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        :worker usually None, networking_interface unless is device
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

        ##
        self.stream_name = stream_name  # this also keeps the subtopic name if using ZMQ
        self.networking_interface = networking_interface
        self.port_number = port_number
        self.data_type = data_type
        # self.preset = get_complete_stream_preset_info(self.stream_name)
        ##

        self.actualSamplingRate = 0

        self.StreamNameLabel.setText(stream_name)
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        self.is_stream_available = False
        self.in_error_state = False  # an error state to prevent ticking when is set to true
        self.is_popped = False

        # visualization data buffer
        self.current_timestamp = 0

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.settings.value('pull_data_interval'))
        self.timer.timeout.connect(self.ticks)

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(int(float(config.settings.value('visualization_refresh_interval'))))
        self.v_timer.timeout.connect(self.visualize)

        # connect btn
        self.StartStopStreamBtn.clicked.connect(self.start_stop_stream_btn_clicked)
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        # inefficient loading of assets TODO need to confirm creating Pixmap in ui_shared result in crash
        self.stream_unavailable_pixmap = QPixmap('../media/icons/streamwidget_stream_unavailable.png')
        self.stream_available_pixmap = QPixmap('../media/icons/streamwidget_stream_available.png')
        self.stream_active_pixmap = QPixmap('../media/icons/streamwidget_stream_viz_active.png')

        # visualization component
        # This variable stores all the visualization components we initialize it in the init_stream_visualization()
        self.viz_components = None
        self.num_points_to_plot = None

        # self.init_server_client()
        self.group_info = collect_stream_all_groups_info(self.stream_name)

        # data elements
        self.viz_data_buffer = None
        self.create_buffer()

        # if (stream_srate := interface.get_nominal_srate()) != nominal_sampling_rate and stream_srate != 0:
        #     print('The stream {0} found in LAN has sampling rate of {1}, '
        #           'overriding in settings: {2}'.format(lsl_name, stream_srate, nominal_sampling_rate))
        #     config.settings.setValue('NominalSamplingRate', stream_srate)

        # load default settings from settings

        self.worker_thread = QThread(self)
        if self.networking_interface == 'LSL':
            channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')
            self.worker = workers.LSLInletWorker(self.stream_name, channel_names, data_type=data_type, RenaTCPInterface=None)
        elif self.networking_interface == 'ZMQ':
            self.worker = workers.ZMQWorker(port_number=port_number, subtopic=stream_name, data_type=data_type)
        elif self.networking_interface == 'Device':
            assert worker
            self.worker = worker
        self.worker.signal_data.connect(self.process_stream_data)
        self.worker.signal_stream_availability.connect(self.update_stream_availability)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # create option window
        self.stream_options_window = StreamOptionsWindow(parent_stream_widget=self, stream_name=self.stream_name,
                                                         group_info=self.group_info, plot_format_changed_signal=self.plot_format_changed_signal)
        self.plot_format_changed_signal.connect(self.plot_format_on_change)
        # self.stream_options_window.plot_format_on_change_signal.connect(self.plot_format_on_change)
        self.stream_options_window.image_change_signal.connect(self.image_changed)

        self.stream_options_window.bar_chart_range_on_change_signal.connect(self.bar_chart_range_on_change)
        self.stream_options_window.hide()

        # create visualization component, must be after the option window
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        self.create_visualization_component()

        # FPS counter``
        self.tick_times = deque(maxlen=10 * int(float(config.settings.value('visualization_refresh_interval'))))

        # mutex for not update the settings while plotting
        self.setting_update_viz_mutex = QMutex()

        self.set_button_icons()
        # start the timers
        self.timer.start()
        self.v_timer.start()

        # Attributes purely for performance checks x############################
        """
        These attributes should be kept only on this perforamnce branch
        """
        self.update_buffer_times = None
        self.plot_data_times = None
        ########################################################################

    def reset_performance_measures(self):
        self.update_buffer_times = []
        self.plot_data_times = []

    def update_stream_availability(self, is_stream_available):
        '''
        this function check if the stream is available
        '''
        print('Stream {0} availability is {1}'.format(self.stream_name, is_stream_available), end='\r')
        self.is_stream_available = is_stream_available
        if self.worker.is_streaming:
            if is_stream_available:
                if not self.StartStopStreamBtn.isEnabled(): self.StartStopStreamBtn.setEnabled(True)
                self.StreamAvailablilityLabel.setPixmap(self.stream_active_pixmap)
                self.StreamAvailablilityLabel.setToolTip("Stream {0} is being plotted".format(self.stream_name))
            else:
                self.start_stop_stream_btn_clicked()  # must stop the stream before dialog popup
                self.set_stream_unavailable()
                dialog_popup('Lost connection to {0}'.format(self.stream_name), title='Warning', mode='modeless')
        else:
            # is the stream is not available
            if is_stream_available:
                self.set_stream_available()
            else:
                self.set_stream_unavailable()
        self.main_parent.update_active_streams()

    def set_stream_unavailable(self):
        self.StartStopStreamBtn.setEnabled(False)
        self.StreamAvailablilityLabel.setPixmap(self.stream_unavailable_pixmap)
        self.StreamAvailablilityLabel.setToolTip("Stream {0} is not available".format(self.stream_name))

    def set_stream_available(self):
        self.StartStopStreamBtn.setEnabled(True)
        self.StreamAvailablilityLabel.setPixmap(self.stream_available_pixmap)
        self.StreamAvailablilityLabel.setToolTip("Stream {0} is available to start".format(self.stream_name))

    def set_button_icons(self):
        if not self.is_streaming():
            self.StartStopStreamBtn.setIcon(start_stream_icon)
        else:
            self.StartStopStreamBtn.setIcon(stop_stream_icon)

        if not self.is_popped:
            self.PopWindowBtn.setIcon(pop_window_icon)
        else:
            self.PopWindowBtn.setIcon(dock_window_icon)

    def options_btn_clicked(self):
        print("Option window open")
        self.stream_options_window.show()
        self.stream_options_window.activateWindow()

    def is_streaming(self):
        return self.worker.is_streaming

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.worker.is_streaming:
            self.worker.stop_stream()
            if not self.worker.is_streaming:
                # started
                # print("sensor stopped")
                self.StartStopStreamBtn.setText("Start Stream")  # toggle the icon
                self.update_stream_availability(self.worker.is_stream_available)
        else:
            try:
                self.worker.start_stream()
            except LSLStreamNotFoundError as e:
                dialog_popup(msg=str(e), title='ERROR')
                return
            except ChannelMismatchError as e:  # only LSL's channel mismatch can be checked at this time, zmq's channel mismatch can only be checked when receiving data
                reply = QMessageBox.question(self, 'Channel Mismatch',
                                             'The stream with name {0} found on the network has {1}.\n'
                                             'The preset has {2} channels. \n '
                                             'Do you want to reset your preset to a default and start stream.\n'
                                             'You can edit your stream channels in Options if you choose No'.format(
                                                 self.stream_name, e.message,
                                                 len(get_stream_preset_info(self.stream_name, 'ChannelNames'))),
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.reset_preset_by_num_channels(e.message)
                    try:
                        self.worker.start_stream()  # start the stream again with updated preset
                    except LSLStreamNotFoundError as e:
                        dialog_popup(msg=str(e), title='ERROR')
                        return
                else:
                    return
            except Exception as e:
                raise UnsupportedErrorTypeError(str(e))
            self.reset_performance_measures()
            if self.worker.is_streaming:
                self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()
        self.main_parent.update_active_streams()

    def reset_preset_by_num_channels(self, num_channels):
        remove_stream_preset_from_settings(self.stream_name)
        self.main_parent.create_preset(self.stream_name, self.data_type, self.port_number, self.networking_interface, num_channels=num_channels)  # update preset in settings
        self.create_buffer()  # recreate the interface and buffer, using the new preset
        self.worker.reset_interface(self.stream_name, get_stream_preset_info(self.stream_name, 'ChannelNames'))

        self.group_info = collect_stream_all_groups_info(self.stream_name)  # get again the group info
        self.stream_options_window.reload_preset_to_UI(self.group_info)
        self.reset_viz()

    def reset_viz(self):
        """
        caller to this function must ensure self.group_info is modified and up to date with user changes
        """
        self.clear_stream_visualizations()
        self.create_visualization_component()

    def create_buffer(self):
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')
        buffer_size = 1 if len(channel_names) > config.MAX_TIMESERIES_NUM_CHANNELS_PER_STREAM else config.VIZ_DATA_BUFFER_MAX_SIZE
        self.viz_data_buffer = DataBufferSingleStream(num_channels=len(channel_names),
                                                      buffer_sizes=buffer_size, append_zeros=True)

    def dock_window(self):
        self.parent.insertWidget(self.parent.count() - 1, self)
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.pop_window)
        # self.PopWindowBtn.setText('Pop Window')
        self.main_parent.pop_windows[self.stream_name].hide()  # tetentive measures
        self.main_parent.pop_windows.pop(self.stream_name)
        self.main_parent.activateWindow()
        self.is_popped = False
        self.set_button_icons()

    def pop_window(self):
        w = AnotherWindow(self, self.remove_stream)
        self.main_parent.pop_windows[self.stream_name] = w
        w.setWindowTitle(self.stream_name)
        # self.PopWindowBtn.setText('Dock Window')
        w.show()
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.dock_window)
        self.is_popped = True
        self.set_button_icons()

    def remove_stream(self):
        self.timer.stop()
        self.v_timer.stop()
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
        # stop_stream_btn.click()  # fire stop streaming first
        if self.worker.is_streaming:
            self.worker.stop_stream()
        # if self.lsl_worker.dsp_on:
        #     self.lsl_worker.remove_stream()
        self.worker_thread.exit()
        self.worker_thread.wait()  # wait for the thread to exit

        # self.main_parent.lsl_workers.pop(self.stream_name)
        # self.main_parent.worker_threads.pop(self.stream_name)
        # if this lsl connect to a device:

        # TODO: we do not consider device at this stage
        # if self.stream_name in self.main_parent.device_workers.keys():
        #     self.main_parent.device_workers[self.stream_name].stop_stream()
        #     self.main_parent.device_workers.pop(self.stream_name)

        self.main_parent.stream_widgets.pop(self.stream_name)
        self.main_parent.remove_stream_widget(self)
        # close window if popped
        if self.stream_name in self.main_parent.pop_windows.keys():
            self.main_parent.pop_windows[self.stream_name].hide()
            # self.main_parent.pop_windows.pop(self.stream_name)
            self.deleteLater()
        else:  # use recursive delete if docked
            self.deleteLater()
        # self.main_parent.LSL_data_buffer_dicts.pop(self.stream_name)
        self.stream_options_window.close()
        # close the signal option window
        return True

    def update_channel_shown(self, channel_index, is_shown, group_name):
        channel_plot_widget = self.channel_index_plot_widget_dict[channel_index]
        channel_plot_widget.show() if is_shown else channel_plot_widget.hide()
        self.group_info = collect_stream_all_groups_info(self.stream_name)  # just reload the group info from settings
        self.update_groups_shown(group_name)

    def update_groups_shown(self, group_name):
        # assuming group info is update to date with in the persist settings
        # check if there's active channels in this group
        if is_group_shown(group_name, self.stream_name):
            self.group_name_plot_widget_dict[group_name].show()
        else:
            self.group_name_plot_widget_dict[group_name].hide()

    def clear_stream_visualizations(self):
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        clear_layout(self.viz_group_layout)

    def init_stream_visualization(self):

        # init stream view with LSL
        # time_series_widget = self.TimeSeriesPlotsLayout

        # fs_label = QLabel(text='Sampling rate = ')
        # ts_label = QLabel(text='Current Time Stamp = ')
        # self.MetaInfoVerticalLayout.addWidget(fs_label)
        # self.MetaInfoVerticalLayout.addWidget(ts_label)
        # if plot_group_slices:
        time_series_widgets = {}
        image_labels = {}
        barchart_widgets = {}
        plots = []
        plot_elements = {}

        # plot_formats = []
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')

        group_plot_widget_dict = {}
        for group_name in self.group_info.keys():
            if is_only_image_enabled := self.group_info[group_name]['is_image_only']:
                update_selected_plot_format(self.stream_name, group_name, 1)  # change the plot format to image now
                self.group_info = collect_stream_all_groups_info(self.stream_name)  # reload the group info from settings

            group_channel_names = [channel_names[int(i)] for i in self.group_info[group_name]['channel_indices']]
            group_plot_widget_dict[group_name] = GroupPlotWidget(self, self.stream_name, group_name, self.group_info[group_name], group_channel_names, get_stream_preset_info(self.stream_name, 'NominalSamplingRate'), self.plot_format_changed_signal)
            self.viz_group_layout.addWidget(group_plot_widget_dict[group_name])
            self.num_points_to_plot = self.get_num_points_to_plot()

        return group_plot_widget_dict

    # def get_viz_time_vector(self):
    #     display_duration = get_stream_preset_info(self.stream_name, 'DisplayDuration')
    #     num_points_to_plot = int(display_duration * get_stream_preset_info(self.stream_name, 'NominalSamplingRate'))
    #     return np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'), num_points_to_plot)

    def create_visualization_component(self):
        group_plot_dict = self.init_stream_visualization()
        self.viz_components = VizComponents(self.fs_label, self.ts_label, group_plot_dict)

    def process_stream_data(self, data_dict):
        '''
        update the visualization buffer, recording buffer, and scripting buffer
        '''
        if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
            try:
                self.update_buffer_times.append(timeit(self.viz_data_buffer.update_buffer, (data_dict, )))  # NOTE performance test scripts, don't include in production code
                # self.viz_data_buffer.update_buffer(data_dict)
            except ChannelMismatchError as e:
                self.in_error_state = True
                reply = QMessageBox.question(self, 'Channel Mismatch',
                                             'The stream with name {0} found on the network has {1}.\n'
                                             'The preset has {2} channels. \n '
                                             'Do you want to reset your preset to a default and start stream.\n'
                                             'If you choose No, streaming will stop. You can edit your stream channels in Options'.format(
                                                 self.stream_name, e.message,
                                                 len(get_stream_preset_info(self.stream_name, 'ChannelNames'))),
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.reset_preset_by_num_channels(e.message)
                    self.in_error_state = False
                    return
                else:
                    self.start_stop_stream_btn_clicked()  # stop the stream
                    self.in_error_state = False
                    return
            self.actualSamplingRate = data_dict['sampling_rate']
            self.current_timestamp = data_dict['timestamps'][-1]
            # notify the internal buffer in recordings tab

            # reshape data_dict based on sensor interface
            self.main_parent.recording_tab.update_recording_buffer(data_dict)
            self.main_parent.scripting_tab.forward_data(data_dict)
            # scripting tab
            # self.main_parent.inference_tab.update_buffers(data_dict)

    '''
    settings on change:
    visualization can be changed while recording with mutex
    1. lock settings on change
    2. update visualization
    3. save changes to RENA_Settings
    4. release mutex
        
    data processing cannot be changed while recording
    
    # cannot add channels while streaming/recording
    
    
    '''

    def stream_settings_changed(self, change):
        self.setting_update_viz_mutex.lock()
        # resolve the
        if change[0] == "nominal_sampling_rate":
            pass  # TODO
        # TODO add other changes such as plot format, plot order, etc...

        self.setting_update_viz_mutex.unlock()

    def visualize(self):
        '''
        This is the function for LSL data visualization.
        It plot the data from the data visualization buffer based on the configuration
        The data to plot is in the parameter self.viz_data_buffer
        '''

        self.tick_times.append(time.time())
        # print("Viz FPS {0}".format(self.get_fps()), end='\r')
        self.worker.signal_stream_availability_tick.emit()  # signal updating the stream availability
        # for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
        actual_sampling_rate = self.actualSamplingRate
        # max_display_datapoint_num = self.stream_widget_visualization_component.plot_widgets[0].size().width()

        # reduce the number of points to plot to the number of pixels in the corresponding plot widget

        # if data_to_plot.shape[-1] > config.DOWNSAMPLE_MULTIPLY_THRESHOLD * max_display_datapoint_num:
        #     data_to_plot = np.nan_to_num(data_to_plot, nan=0)
        #     # start = time.time()
        #     # data_to_plot = data_to_plot[:, ::int(data_to_plot.shape[-1] / max_display_datapoint_num)]
        #     # data_to_plot = signal.resample(data_to_plot, int(data_to_plot.shape[-1] / max_display_datapoint_num), axis=1)
        #     data_to_plot = decimate(data_to_plot, q=int(data_to_plot.shape[-1] / max_display_datapoint_num),
        #                             axis=1)  # resample to 100 hz with retain history of 10 sec
        #     # print(time.time()-start)
        #     time_vector = np.linspace(0., config.PLOT_RETAIN_HISTORY, num=data_to_plot.shape[-1])

        # self.LSL_plots_fs_label_dict[lsl_stream_name][2].setText(
        #     'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        #
        # [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
        #  enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]
        # change to loop with type condition plot time_series and image
        # if self.LSL_plots_fs_label_dict[lsl_stream_name][3]:
        # plot_channel_num_offset = 0
        if not self.viz_data_buffer.has_data():
            return
        data_to_plot = self.viz_data_buffer.buffer[0][:, -self.num_points_to_plot:]

        for plot_group_index, (group_name) in enumerate(self.group_info.keys()):
            self.viz_components.group_plots[group_name].plot_data(data_to_plot)

        # show the label
        self.viz_components.fs_label.setText(
            'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.viz_components.ts_label.setText(
            'Current Time Stamp = {0}'.format(self.current_timestamp))

    def ticks(self):
        self.worker.signal_data_tick.emit()
        #     self.recent_tick_refresh_timestamps.append(time.time())
        #     if len(self.recent_tick_refresh_timestamps) > 2:
        #         self.tick_rate = 1 / ((self.recent_tick_refresh_timestamps[-1] - self.recent_tick_refresh_timestamps[0]) / (
        #                     len(self.recent_tick_refresh_timestamps) - 1))
        #
        #     self.tickFrequencyLabel.setText(
        #         'Pull Data Frequency: {0}'.format(round(self.tick_rate, config_ui.tick_frequency_decimal_places)))

    def init_server_client(self):
        print('John')

        # dummy preset for now:
        stream_name = 'OpenBCI'
        port_id = int(time.time())
        identity = 'server'
        processor_dic = {}
        rena_tcp_request_object = RenaTCPAddDSPWorkerRequestObject(stream_name, port_id, identity, processor_dic)
        self.main_parent.rena_dsp_client.send_obj(rena_tcp_request_object)
        rena_tcp_request_object = self.main_parent.rena_dsp_client.recv_obj()
        print('DSP worker created')
        self.dsp_client_interface = RenaTCPInterface(stream_name=stream_name, port_id=port_id, identity='client')

        # send to server

    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0

    def is_widget_streaming(self):
        return self.worker.is_streaming

    def on_num_points_to_display_change(self, num_points_to_plot, new_sampling_rate, new_display_duration):
        '''
        this function is called by StreamOptionWindow when user change number of points to plot by changing
        the sampling rate or the display duration
        :param num_points_to_plot:
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        self.update_sr_and_display_duration_in_settings(new_sampling_rate, new_display_duration)
        self.num_points_to_plot = self.get_num_points_to_plot()

    def update_sr_and_display_duration_in_settings(self, new_sampling_rate, new_display_duration):
        '''
        this function is called by StreamWidget when on_num_points_to_display_change is called
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        set_stream_preset_info(self.stream_name, 'NominalSamplingRate', new_sampling_rate)
        set_stream_preset_info(self.stream_name, 'DisplayDuration', new_display_duration)

    # def reload_visualization_elements(self, info_dict):
    #     self.group_info = collect_stream_all_groups_info(self.stream_name)
    #     clear_layout(self.MetaInfoVerticalLayout)
    #     clear_layout(self.TimeSeriesPlotsLayout)
    #     clear_layout(self.ImageWidgetLayout)
    #     clear_layout(self.BarPlotWidgetLayout)
    #     self.create_visualization_component()

    @QtCore.pyqtSlot(dict)
    def plot_format_on_change(self, info_dict):
        # old_format = self.group_info[info_dict['group_name']]['selected_plot_format']
        self.group_info[info_dict['group_name']]['selected_plot_format'] = info_dict['new_format']

        # self.preset_on_change()  # update the group info

        # self.viz_components.group_plots[plot_format_index_dict[old_format]][
        #     info_dict['group_name']].hide()
        # self.viz_components.group_plots[plot_format_index_dict[info_dict['new_format']]][
        #     info_dict['group_name']].show()

        # update the plot hide display

    @QtCore.pyqtSlot(dict)
    def image_changed(self, change: dict):
        self.group_info[change['group_name']]['plot_format']['image'] = change['this_group_info_image']
        if change['group_name'] in self.group_name_plot_widget_dict.keys():
            self.group_name_plot_widget_dict[change['group_name']].update_image_info(change['this_group_info_image'])

    def preset_on_change(self):
        self.group_info = collect_stream_all_groups_info(self.stream_name)  # reload the group info

    def get_image_format_and_shape(self, group_name):
        width = self.group_info[group_name]['plot_format']['image']['width']
        height = self.group_info[group_name]['plot_format']['image']['height']
        image_format = self.group_info[group_name]['plot_format']['image']['image_format']
        depth = image_depth_dict[image_format]
        channel_format = self.group_info[group_name]['plot_format']['image']['channel_format']
        scaling_factor = self.group_info[group_name]['plot_format']['image']['scaling_factor']

        return width, height, depth, image_format, channel_format, scaling_factor

    #############################################

    def bar_chart_range_on_change(self, group_name, barchart_min, barchart_max):
        self.group_info[group_name]['plot_format']['bar_chart']['y_min'] = barchart_min
        self.group_info[group_name]['plot_format']['bar_chart']['y_max'] = barchart_max
        self.viz_components.group_plots[group_name].update_bar_chart_range(self.group_info[group_name])

    #############################################

    def channel_group_changed(self, change_dict):
        """
        Called when one or more channel's parent group is changed
        @param change_dict:
        """
        # update the group info
        for group_name, child_channels in change_dict.items():
            if len(child_channels) == 0:
                self.group_info.pop(group_name)
            else:  # cover the cases for both changed groups and new group
                if group_name not in self.group_info.keys():
                    self.group_info[group_name] = create_default_group_info(len(child_channels), group_name)[group_name]  # add a default group info
                self.group_info[group_name]['channel_indices'] = [x.lsl_index for x in child_channels]
                self.group_info[group_name]['is_channels_shown'] = [int(x.is_shown) for x in child_channels]
        export_group_info_to_settings(self.group_info, self.stream_name)
        self.reset_viz()

    def change_group_name(self, new_group_name, old_group_name):
        self.group_info[new_group_name] = self.group_info.pop(old_group_name)
        self.viz_components.group_plots[new_group_name] = self.viz_components.group_plots.pop(old_group_name)

        self.viz_components.group_plots[new_group_name].change_group_name(new_group_name)
        change_group_name(self.group_info[new_group_name], new_group_name, old_group_name, self.stream_name)

    def change_channel_name(self, group_name, new_ch_name, old_ch_name, lsl_index):
        # change channel name in the settings
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')
        changing_channel_index = channel_names.index(old_ch_name)
        channel_names[changing_channel_index] = new_ch_name
        set_stream_preset_info(self.stream_name, 'ChannelNames', channel_names)

        # change the name in the plots
        self.viz_components.group_plots[group_name].change_channel_name(new_ch_name, old_ch_name, lsl_index)

    def get_num_points_to_plot(self):
        display_duration = get_stream_preset_info(self.stream_name, 'DisplayDuration')
        return int(display_duration * get_stream_preset_info(self.stream_name, 'NominalSamplingRate'))

    # def add_group(self, affected):
    #     new_group_name = self.get_next_available_groupname()
    #     print("StreamOptionsWindow: creating new group {}".format(new_group_name))
    #
    #     for group_name, child_channels in affected.items():
    #         self.group_info
    #         if len(child_channels) == 0:
    #             self.group_info.pop(group_name)
    #         else:
    #             self.group_info[group_name]['channel_indices'] = [x.lsl_index for x in child_channels]
    #             self.group_info[group_name]['is_channels_shown'] = [int(x.is_shown) for x in child_channels]
    #     return new_group_name

    def get_pull_data_delay(self):
        return self.worker.get_pull_data_delay()