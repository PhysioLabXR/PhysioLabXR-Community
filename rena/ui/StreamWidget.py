# This Python file uses the following encoding: utf-8
import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer, QThread, QMutex
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox
from pyqtgraph import PlotDataItem

from exceptions.exceptions import RenaError, LSLChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from rena import config, config_ui
from rena.config_ui import image_depth_dict
from rena.sub_process.TCPInterface import RenaTCPAddDSPWorkerRequestObject, RenaTCPInterface
from rena.interfaces.LSLInletInterface import LSLInletInterface
from rena.threadings import workers
from rena.ui.StreamOptionsWindow import StreamOptionsWindow
from rena.ui.StreamWidgetVisualizationComponents import StreamWidgetVisualizationComponents
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
    options_icon
from rena.utils.general import create_lsl_interface, DataBufferSingleStream
from rena.utils.settings_utils import get_childKeys_for_group, get_childGroups_for_group, get_stream_preset_info, \
    collect_stream_all_groups_info, get_complete_stream_preset_info, is_group_shown, remove_stream_preset_from_settings, \
    create_default_preset, set_stream_preset_info
from rena.utils.ui_utils import AnotherWindow, dialog_popup, get_distinct_colors, clear_layout


class StreamWidget(QtWidgets.QWidget):
    def __init__(self, main_parent, parent, stream_name, insert_position=None):
        """
        LSL interface is created in StreamWidget
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
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
        self.stream_name = stream_name
        # self.preset = get_complete_stream_preset_info(self.stream_name)
        ##

        self.actualSamplingRate = 0

        self.StreamNameLabel.setText(stream_name)
        self.set_button_icons()
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        self.is_stream_available = False

        # visualization data buffer
        self.current_timestamp = 0

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 1000 Hz refresh rate
        self.timer.timeout.connect(self.ticks)

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        self.v_timer.timeout.connect(self.visualize_LSLStream_data)

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
        self.stream_widget_visualization_component = None

        # self.init_server_client()

        # data elements
        self.interface, self.viz_data_buffer = None, None
        self.create_interface_and_buffer()

        # if (stream_srate := interface.get_nominal_srate()) != nominal_sampling_rate and stream_srate != 0:
        #     print('The stream {0} found in LAN has sampling rate of {1}, '
        #           'overriding in settings: {2}'.format(lsl_name, stream_srate, nominal_sampling_rate))
        #     config.settings.setValue('NominalSamplingRate', stream_srate)

        # load default settings from settings

        self.worker_thread = QThread(self)
        self.lsl_worker = workers.LSLInletWorker(LSLInlet_interface=self.interface,
                                                 RenaTCPInterface=None)
        self.lsl_worker.signal_data.connect(self.process_LSLStream_data)
        self.lsl_worker.signal_stream_availability.connect(self.update_stream_availability)
        self.lsl_worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # create visualization component:
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        self.group_info = collect_stream_all_groups_info(self.stream_name)
        self.viz_time_vector = None
        self.create_visualization_component()

        # create option window
        self.stream_options_window = StreamOptionsWindow(parent=self, stream_name=self.stream_name,
                                                         group_info=self.group_info)
        self.stream_options_window.hide()

        # FPS counter``
        self.tick_times = deque(maxlen=config.VISUALIZATION_REFRESH_INTERVAL)

        # mutex for not update the settings while plotting
        self.setting_update_viz_mutex = QMutex()

        # start the timers
        self.timer.start()
        self.v_timer.start()

    def update_stream_availability(self, is_stream_available):
        print('Stream {0} availability is {1}'.format(self.stream_name, is_stream_available), end='\r')
        self.is_stream_available = is_stream_available
        if self.lsl_worker.is_streaming:
            if is_stream_available:
                if not self.StartStopStreamBtn.isEnabled(): self.StartStopStreamBtn.setEnabled(True)
                self.StreamAvailablilityLabel.setPixmap(self.stream_active_pixmap)
                self.StreamAvailablilityLabel.setToolTip("Stream {0} is being plotted".format(self.stream_name))
            else:
                self.start_stop_stream_btn_clicked()  # must stop the stream before dialog popup
                self.set_stream_unavailable()
                dialog_popup('Lost connection to {0}'.format(self.stream_name), title='Warning')
        else:
            # is the stream is not available
            if is_stream_available:
                self.set_stream_available()
            else:
                self.set_stream_unavailable()
        self.main_parent.update_num_active_stream_label()

    def set_stream_unavailable(self):
        self.StartStopStreamBtn.setEnabled(False)
        self.StreamAvailablilityLabel.setPixmap(self.stream_unavailable_pixmap)
        self.StreamAvailablilityLabel.setToolTip("Stream {0} is not available".format(self.stream_name))

    def set_stream_available(self):
        self.StartStopStreamBtn.setEnabled(True)
        self.StreamAvailablilityLabel.setPixmap(self.stream_available_pixmap)
        self.StreamAvailablilityLabel.setToolTip("Stream {0} is available to start".format(self.stream_name))

    def set_button_icons(self):
        if 'Start' in self.StartStopStreamBtn.text():
            self.StartStopStreamBtn.setIcon(start_stream_icon)
        else:
            self.StartStopStreamBtn.setIcon(stop_stream_icon)

        if 'Pop' in self.PopWindowBtn.text():
            self.PopWindowBtn.setIcon(pop_window_icon)
        else:
            self.PopWindowBtn.setIcon(dock_window_icon)

    def options_btn_clicked(self):
        print("Option window open")
        self.stream_options_window.show()
        self.stream_options_window.activateWindow()

    def is_streaming(self):
        return self.lsl_worker.is_streaming

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.lsl_worker.is_streaming:
            self.lsl_worker.stop_stream()
            if not self.lsl_worker.is_streaming:
                # started
                print("sensor stopped")
                self.StartStopStreamBtn.setText("Start Stream")  # toggle the icon
                self.update_stream_availability(self.lsl_worker.is_stream_available)
        else:
            try:
                self.lsl_worker.start_stream()
            except LSLStreamNotFoundError as e:
                dialog_popup(msg=str(e), title='ERROR')
                return
            except LSLChannelMismatchError as e:
                reply = QMessageBox.question(self, 'Channel Mismatch', 'The LSL stream with name {0} found on the network has {1}.\n'
                                                                       'The preset has {2} channels. \n '
                                                                       'Do you want to reset your preset to a default and start stream.\n'
                                                                       'You can edit your stream channels in Options if you choose No'.format(self.stream_name, e.message, len(get_stream_preset_info(self.stream_name, 'ChannelNames'))),
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.reset_preset_by_num_channels(e.message)
                    self.lsl_worker.start_stream()  # start the stream again with updated preset
                else: return
            except Exception as e:
                raise UnsupportedErrorTypeError(str(e))

            if self.lsl_worker.is_streaming:
                # started
                print("sensor stopped")
                # toggle the icon
                self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()
        self.main_parent.update_num_active_stream_label()

    def reset_preset_by_num_channels(self, num_channels):
        remove_stream_preset_from_settings(self.stream_name)
        create_default_preset(self.stream_name, num_channels=num_channels)  # update preset in settings
        self.create_interface_and_buffer()  # recreate the interface and buffer, using the new preset
        self.lsl_worker.set_interface(self.interface)
        self.stream_options_window.reload_preset_to_UI()
        self.clear_stream_visualizations()
        self.create_visualization_component()

    def create_interface_and_buffer(self):
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')
        self.interface = create_lsl_interface(self.stream_name, channel_names)  # maybe want to discard and close the old interface
        self.viz_data_buffer = DataBufferSingleStream(num_channels=len(channel_names), buffer_sizes=config.VIZ_DATA_BUFFER_MAX_SIZE, append_zeros=True)

    def dock_window(self):
        self.parent.insertWidget(self.parent.count() - 1, self)
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.PopWindowBtn.setText('Pop Window')
        self.main_parent.pop_windows[self.stream_name].hide()  # tetentive measures
        self.main_parent.pop_windows.pop(self.stream_name)
        self.set_button_icons()
        self.main_parent.activateWindow()

    def pop_window(self):
        w = AnotherWindow(self, self.remove_stream)
        self.main_parent.pop_windows[self.stream_name] = w
        w.setWindowTitle(self.stream_name)
        self.PopWindowBtn.setText('Dock Window')
        w.show()
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.dock_window)
        self.set_button_icons()

    def remove_stream(self):
        if self.main_parent.recording_tab.is_recording:
            dialog_popup(msg='Cannot remove stream while recording.')
            return False
        # stop_stream_btn.click()  # fire stop streaming first
        if self.lsl_worker.is_streaming:
            self.lsl_worker.stop_stream()
        # if self.lsl_worker.dsp_on:
        #     self.lsl_worker.remove_stream()
        self.worker_thread.exit()
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
        self.group_info = collect_stream_all_groups_info(self.stream_name)  # get again the group info
        self.viz_time_vector = None
        clear_layout(self.TimeSeriesPlotsLayout)
        clear_layout(self.ImageWidgetLayout)
        clear_layout(self.MetaInfoVerticalLayout)

    def init_stream_visualization(self):

        # init stream view with LSL
        # time_series_widget = self.TimeSeriesPlotsLayout

        fs_label = QLabel(text='Sampling rate = ')
        ts_label = QLabel(text='Current Time Stamp = ')
        self.MetaInfoVerticalLayout.addWidget(fs_label)
        self.MetaInfoVerticalLayout.addWidget(ts_label)
        # if plot_group_slices:
        plot_widgets = {}
        plots = []
        # plot_formats = []
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')
        for group_name in self.group_info.keys():
            plot_format = self.group_info[group_name]['plot_format']
            # one plot widget for each group, no need to check chan_names because plot_group_slices only comes with preset
            if plot_format['time_series']['display']:  # time_series plot
                # plot_formats.append(plot_group_format_info)
                group_plot_widget = pg.PlotWidget()
                self.group_name_plot_widget_dict[group_name] = group_plot_widget
                self.TimeSeriesPlotsLayout.addWidget(group_plot_widget)

                distinct_colors = get_distinct_colors(len(self.group_info[group_name]['channel_indices']))
                group_plot_widget.addLegend()

                plot_data_items = []
                group_channel_names = [channel_names[int(i)] for i in self.group_info[group_name]['channel_indices']]  # channel names for this group
                for channel_index_in_group, (channel_index, channel_name) in enumerate(zip(self.group_info[group_name]['channel_indices'], group_channel_names)):
                    channel_plot_widget = group_plot_widget.plot([], [], pen=pg.mkPen(color=distinct_colors[channel_index_in_group]),  # unique color for each group
                                         name=channel_name)
                    self.channel_index_plot_widget_dict[int(channel_index)] = channel_plot_widget
                    plot_data_items.append(channel_plot_widget)
                    if not self.group_info[group_name]['is_channels_shown'][channel_index_in_group]:  # if this channel is not shown
                        channel_plot_widget.hide()
                self.update_groups_shown(group_name)
                plots.append(plot_data_items)
                plot_widgets[group_name] = group_plot_widget

            if plot_format['image']['display']:
                image_shape = [plot_format['image']['width'], plot_format['image']['width'], image_depth_dict[plot_format['image']['format']]]
                image_label = QLabel('Image_Label')
                self.ImageWidgetLayout.addWidget(image_label)

                # image_label.setAlignment(QtCore.Qt.AlignCenter)
                # pass

            if plot_format['bar_plot']['display']:
                pass

                # elif plot_group_format_info[0] == 'image':
                #     plot_group_format_info[1] = tuple(eval(plot_group_format_info[1]))
                #
                #     # check if the channel num matches:
                #     if pg_slice[1] - pg_slice[0] != np.prod(np.array(plot_group_format_info[1])):
                #         raise AssertionError(
                #             'The number of channel in this slice does not match with the number of image pixels.'
                #             'The image format is {0} but channel slice format is {1}'.format(plot_group_format_info,
                #                                                                              pg_slice))
                #     plot_formats.append(plot_group_format_info)
                #     image_label = QLabel('Image_Label')
                #     image_label.setAlignment(QtCore.Qt.AlignCenter)
                #     parent.addWidget(image_label)
                #     plots.append(image_label)
                # else:
                #     raise AssertionError('Unknown plotting group format. We only support: time_series, image_(a,b,c)')

        [p.setDownsampling(auto=True, method='mean') for group in plots for p in group if p is PlotDataItem]
        [p.setClipToView(clip=True) for p in plots for group in plots for p in group if p is PlotDataItem]

        self.viz_time_vector = self.get_viz_time_vector()
        return fs_label, ts_label, plot_widgets, plots

    def get_viz_time_vector(self):
        display_duration = get_stream_preset_info(self.stream_name, 'DisplayDuration')
        num_points_to_plot = int(display_duration * get_stream_preset_info(self.stream_name, 'NominalSamplingRate'))
        return np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'), num_points_to_plot)

    def create_visualization_component(self):
        fs_label, ts_label, plot_widgets, plots = \
            self.init_stream_visualization()
        self.stream_widget_visualization_component = \
            StreamWidgetVisualizationComponents(fs_label, ts_label, plot_widgets, plots)

    def process_LSLStream_data(self, data_dict):
        if data_dict['frames'].shape[-1] > 0:  # if there are data in the emited data dict
            self.viz_data_buffer.update_buffer(data_dict)

            # buffered_data = self.viz_data_buffer
            # try:
            #     buffered_data = np.concatenate(
            #         (buffered_data, data_dict['frames']),
            #         axis=-1)  # get all data and remove it from internal buffer
            #     self.current_timestamp = data_dict['timestamps'][-1]
            # except ValueError:
            #     raise Exception('The number of channels for stream {0} mismatch from its preset json.'.format(
            #         data_dict['lsl_data_type']))
            # if buffered_data.shape[-1] < self.num_samples_to_plot:
            #     data_to_plot = np.concatenate((np.zeros(shape=(
            #         buffered_data.shape[0],
            #         self.num_samples_to_plot -
            #         buffered_data.shape[-1])),
            #                                    buffered_data), axis=-1)
            # else:
            #     data_to_plot = buffered_data[:,
            #                    - self.num_samples_to_plot:]  # plot the most recent few seconds

            # main window only retains the most recent 10 seconds for visualization purposes
            self.actualSamplingRate = data_dict['sampling_rate']
            # notify the internal buffer in recordings tab

            # reshape data_dict based on sensor interface
            self.main_parent.recording_tab.update_buffers(data_dict)
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

    def visualize_LSLStream_data(self):
        self.tick_times.append(time.time())
        print("Viz FPS {0}".format(self.get_fps()), end='\r')
        self.lsl_worker.signal_stream_availability_tick.emit()  # signal updating the stream availability
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
        data_to_plot = self.viz_data_buffer.buffer[0][:, -len(self.viz_time_vector):]
        for plot_group_index, (group_name) in enumerate(self.group_info.keys()):
            plot_group_info = self.group_info[group_name]
            if plot_group_info["plot_format"]['time_series']['display']:
                # plot corresponding time series data, range (a,b) is time series
                # plot_group_channel_num = len(plot_group_info['channels'])
                for index_in_group, channel_index in enumerate(plot_group_info['channel_indices']):
                    if plot_group_info['is_channels_shown'][index_in_group]:
                        # print(channel_index)
                        self.stream_widget_visualization_component.plots[plot_group_index][index_in_group] \
                            .setData(self.viz_time_vector, data_to_plot[int(channel_index), :])
                    # for i in range(plot_channel_num_offset, plot_channel_num_offset + plot_group_channel_num):
                    #     self.LSL_plots_fs_label_dict[lsl_stream_name][0][i].setData(time_vector,
                    #                                                                 data_to_plot[i, :])
                    # plot_channel_num_offset += plot_group_channel_num
                # elif plot_format[0] == 'image':
                #     image_shape = plot_format[1]
                #     channel_num = image_shape[2]
                #     plot_array = data_to_plot[plot_group[0]: plot_group[1], -1]
                #
                #     img = plot_array.reshape(image_shape)
                #     # display openCV image if channel_num = 3
                #     # display heat map if channel_num = 1
                #     if channel_num == 3:
                #         img = convert_cv_qt(img)
                #     if channel_num == 1:
                #         img = np.squeeze(img, axis=-1)
                #         img = convert_heatmap_qt(img)
                #
                #     self.LSL_plots_fs_label_dict[lsl_stream_name][0][plot_channel_num_offset].setPixmap(img)
                #     plot_channel_num_offset += 1

        # TODO： remove this statement
        # else:
        #     [plot.setData(time_vector, data_to_plot[i, :]) for i, plot in
        #      enumerate(self.LSL_plots_fs_label_dict[lsl_stream_name][0])]

        self.stream_widget_visualization_component.fs_label.setText(
            'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.stream_widget_visualization_component.ts_label.setText(
            'Current Time Stamp = {0}'.format(self.current_timestamp))

    # calculate and update the frame rate
    # self.recent_visualization_refresh_timestamps.append(time.time())
    # if len(self.recent_visualization_refresh_timestamps) > 2:
    #     self.visualization_fps = 1 / ((self.recent_visualization_refresh_timestamps[-1] -
    #                                    self.recent_visualization_refresh_timestamps[0]) / (
    #                                               len(self.recent_visualization_refresh_timestamps) - 1))
    # # print("visualization refresh frequency: "+ str(self.visualization_refresh_frequency))
    # # print("John")
    # self.visualizationFPSLabel.setText(
    #     'Visualization FPS: {0}'.format(round(self.visualization_fps, config_ui.visualization_fps_decimal_places)))

    def ticks(self):
        self.lsl_worker.signal_data_tick.emit()
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
        except ZeroDivisionError:
            return 0

    def is_widget_streaming(self):
        return self.lsl_worker.is_streaming

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
        self.viz_time_vector = np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'), num_points_to_plot)

    def update_sr_and_display_duration_in_settings(self, new_sampling_rate, new_display_duration):
        '''
        this function is called by StreamWidget when on_num_points_to_display_change is called
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        set_stream_preset_info(self.stream_name, 'NominalSamplingRate', new_sampling_rate)
        set_stream_preset_info(self.stream_name, 'DisplayDuration', new_display_duration)