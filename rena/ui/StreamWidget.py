# This Python file uses the following encoding: utf-8
import time
from collections import deque
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic, QtCore
from PyQt5.QtCore import QTimer, QThread, QMutex
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox
from pyqtgraph import PlotDataItem

from exceptions.exceptions import RenaError, ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from rena import config, config_ui
from rena.config_ui import image_depth_dict, plot_format_index_dict
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
    create_default_preset, set_stream_preset_info, get_channel_num, collect_stream_group_plot_format, \
    update_selected_plot_format
from rena.utils.ui_utils import AnotherWindow, dialog_popup, get_distinct_colors, clear_layout, \
    convert_array_to_qt_heatmap, \
    convert_rgb_to_qt_image, convert_numpy_to_uint8


class StreamWidget(QtWidgets.QWidget):
    def __init__(self, main_parent, parent, stream_name, data_type, worker, networking_interface, port_number,
                 insert_position=None):
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
        self.set_button_icons()
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        self.is_stream_available = False         # it will automatically detect if the stream is available
        self.in_error_state = False  # an error state to prevent ticking when is set to true

        # visualization data buffer
        self.current_timestamp = 0

        # timer: the rate to pull the data from LSL
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
        # This variable stores all the visualization components we initialize it in the init_stream_visualization()
        self.stream_widget_visualization_component = None # TODO: Drag Drop Rename

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

        # create visualization component:
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        self.viz_time_vector = None
        # self.create_visualization_component()

        # create option window
        self.stream_options_window = StreamOptionsWindow(parent=self, stream_name=self.stream_name,
                                                         group_info=self.group_info)
        self.stream_options_window.plot_format_on_change_signal.connect(self.plot_format_on_change)
        self.stream_options_window.preset_on_change_signal.connect(self.preset_on_change)
        self.stream_options_window.bar_chart_range_on_change_signal.connect(self.bar_chart_range_on_change)
        self.stream_options_window.hide()

        # create visualization component, must be after the option window
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        self.viz_time_vector = None
        self.create_visualization_component()

        # FPS counter``
        self.tick_times = deque(maxlen=config.VISUALIZATION_REFRESH_INTERVAL)

        # mutex for not update the settings while plotting
        self.setting_update_viz_mutex = QMutex()

        # start the timers
        self.timer.start()
        self.v_timer.start()

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
        return self.worker.is_streaming

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.worker.is_streaming:
            self.worker.stop_stream()
            if not self.worker.is_streaming:
                # started
                print("sensor stopped")
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

            if self.worker.is_streaming:
                # started
                print("sensor stopped")
                self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()
        self.main_parent.update_num_active_stream_label()

    def reset_preset_by_num_channels(self, num_channels):
        remove_stream_preset_from_settings(self.stream_name)
        self.main_parent.create_preset(self.stream_name, self.data_type, self.port_number, self.networking_interface, num_channels=num_channels)  # update preset in settings
        self.create_buffer()  # recreate the interface and buffer, using the new preset
        self.worker.reset_interface(self.stream_name, get_stream_preset_info(self.stream_name, 'ChannelNames'))
        self.stream_options_window.reload_preset_to_UI()
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
        time_series_widgets = {}
        image_labels = {}
        barchart_widgets = {}
        plots = []
        plot_elements = {}

        # plot_formats = []
        channel_names = get_stream_preset_info(self.stream_name, 'ChannelNames')

        is_only_image_enabled = False
        for group_name in self.group_info.keys():
            if is_only_image_enabled := self.group_info[group_name]['is_image_only']:
                # disable time series and bar plot for this group
                update_selected_plot_format(self.stream_name, group_name, 1)  # change the plot format to image now
                self.group_info = collect_stream_all_groups_info(self.stream_name)  # reload the group info from settings

            ################################ time series widget initialization###########################################
            if not is_only_image_enabled:
                group_plot_widget = pg.PlotWidget(title=group_name)
                self.group_name_plot_widget_dict[group_name] = group_plot_widget
                self.TimeSeriesPlotsScrollAreaLayout.addWidget(group_plot_widget)

                distinct_colors = get_distinct_colors(len(self.group_info[group_name]['channel_indices']))
                group_plot_widget.addLegend()

                plot_data_items = []
                group_channel_names = [channel_names[int(i)] for i in
                                       self.group_info[group_name]['channel_indices']]  # channel names for this group
                for channel_index_in_group, (channel_index, channel_name) in enumerate(
                        zip(self.group_info[group_name]['channel_indices'], group_channel_names)):
                    if self.group_info[group_name]['is_channels_shown'][
                        channel_index_in_group]:  # if this channel is not shown
                        channel_plot_widget = group_plot_widget.plot([], [], pen=pg.mkPen(
                            color=distinct_colors[channel_index_in_group]),  # unique color for each group
                                                                     name=channel_name)
                        self.channel_index_plot_widget_dict[int(channel_index)] = channel_plot_widget
                        plot_data_items.append(channel_plot_widget)
                        # TODO add back the channel when they are renabled

                self.update_groups_shown(group_name)
                plots.append(plot_data_items)
                time_series_widgets[group_name] = group_plot_widget
                [p.setDownsampling(auto=True, method='mean') for group in plots for p in group if p is PlotDataItem]
                [p.setClipToView(clip=True) for p in plots for group in plots for p in group if p is PlotDataItem]
                if self.group_info[group_name]['selected_plot_format'] != 0:
                    group_plot_widget.hide()

            ############################### init image label ####################################################################
            image_label = QLabel('Image_Label')
            image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.ImageWidgetLayout.addWidget(image_label)
            image_labels[group_name] = image_label
            if self.group_info[group_name]['selected_plot_format'] != 1:
                image_label.hide()

            ############################## bar plot ##############################################################################
            if not is_only_image_enabled:
                barchart_widget = pg.PlotWidget(title=group_name)
                barchart_widget.setYRange(self.group_info[group_name]['plot_format']['bar_chart']['y_min'],
                                          self.group_info[group_name]['plot_format']['bar_chart']['y_max'])
                # barchart_widget.sigRangeChanged.connect(self.bar_chart_range_changed)
                # barchart_widget.setLimits(xMin=-0.5, xMax=len(self.group_info[group_name]['channel_indices']), yMin=plot_format['bar_chart']['y_min'], yMax=plot_format['bar_chart']['y_max'])
                label_x_axis = barchart_widget.getAxis('bottom')
                label_dict = dict(enumerate(group_channel_names)).items()
                label_x_axis.setTicks([label_dict])
                x = np.arange(len(group_channel_names))
                y = np.array([0] * len(group_channel_names))
                bars = pg.BarGraphItem(x=x, height=y, width=1, brush='r')
                barchart_widget.addItem(bars)
                self.BarPlotWidgetLayout.addWidget(barchart_widget)
                barchart_widgets[group_name] = barchart_widget
                if self.group_info[group_name]['selected_plot_format'] != 2:
                    barchart_widget.hide()
        plot_elements['time_series'] = time_series_widgets
        plot_elements['image'] = image_labels
        plot_elements['bar_chart'] = barchart_widgets

        self.viz_time_vector = self.get_viz_time_vector()
        return fs_label, ts_label, plot_elements

    def get_viz_time_vector(self):
        display_duration = get_stream_preset_info(self.stream_name, 'DisplayDuration')
        num_points_to_plot = int(display_duration * get_stream_preset_info(self.stream_name, 'NominalSamplingRate'))
        return np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'), num_points_to_plot)

    def create_visualization_component(self):
        fs_label, ts_label, plot_elements = \
            self.init_stream_visualization()
        self.stream_widget_visualization_component = \
            StreamWidgetVisualizationComponents(fs_label, ts_label, plot_elements)

    def process_stream_data(self, data_dict):
        '''
        update the visualization buffer, recording buffer, and scripting buffer
        '''
        if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
            try:
                self.viz_data_buffer.update_buffer(data_dict)
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

    def visualize_LSLStream_data(self):
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
        data_to_plot = self.viz_data_buffer.buffer[0][:, -len(self.viz_time_vector):]
        for plot_group_index, (group_name) in enumerate(self.group_info.keys()):
            plot_group_info = self.group_info[group_name]
            selected_plot_format = plot_group_info['selected_plot_format']

            # get target plotting
            # plot if valid

            # 1. time_series
            if plot_format_index_dict[selected_plot_format] == 'time_series':
                # plot time series
                if plot_group_info["plot_format"]['time_series']['display']:  # want to show this ?
                    if plot_group_info["plot_format"]['time_series']['is_valid']:  # if the format setting is valid?
                        # plot if valid and display this group
                        for index_in_group, channel_index in enumerate(plot_group_info['channel_indices']):
                            if plot_group_info['is_channels_shown'][index_in_group]:
                                # print(channel_index)
                                self.stream_widget_visualization_component.plot_elements['time_series'][
                                    group_name].plotItem.curves[index_in_group] \
                                    .setData(self.viz_time_vector, data_to_plot[int(channel_index), :])

            # 2. image
            elif plot_format_index_dict[selected_plot_format] == 'image':

                if plot_group_info["plot_format"]['image']['is_valid']:  # if the format setting is valid we continue
                    # reshape and attach to the label
                    width, height, depth, image_format, channel_format, scaling_factor = self.get_image_format_and_shape(
                        group_name)

                    image_plot_data = data_to_plot[
                        plot_group_info['channel_indices'], -1]  # only visualize the last frame

                    # if we chose RGB
                    if image_format == 'RGB':

                        if channel_format == 'Channel First':
                            image_plot_data = np.reshape(image_plot_data, (depth, height, width))
                            image_plot_data = np.moveaxis(image_plot_data, 0, -1)
                        elif channel_format == 'Channel Last':
                            image_plot_data = np.reshape(image_plot_data, (height, width, depth))
                        # image_plot_data = convert_numpy_to_uint8(image_plot_data)
                        image_plot_data = image_plot_data.astype(np.uint8)
                        image_plot_data = convert_rgb_to_qt_image(image_plot_data, scaling_factor=scaling_factor)
                        self.stream_widget_visualization_component.plot_elements['image'][group_name].setPixmap(
                            image_plot_data)

                    # if we chose PixelMap
                    if image_format == 'PixelMap':
                        # pixel map return value
                        image_plot_data = np.reshape(image_plot_data, (height, width))  # matrix : (height, width)
                        image_plot_data = convert_array_to_qt_heatmap(image_plot_data, scaling_factor=scaling_factor)
                        self.stream_widget_visualization_component.plot_elements['image'][group_name].setPixmap(
                            image_plot_data)

            # 3. bar_chart
            elif plot_format_index_dict[selected_plot_format] == 'bar_chart':
                if plot_group_info["plot_format"]['bar_chart']['is_valid']:
                    bar_chart_plot_data = data_to_plot[
                        plot_group_info['channel_indices'], -1]  # only visualize the last frame
                    self.stream_widget_visualization_component.plot_elements['bar_chart'][group_name].plotItem.curves[
                        0].setOpts(x=np.arange(len(bar_chart_plot_data)), height=bar_chart_plot_data, width=1,
                                   brush='r')

        # show the label
        self.stream_widget_visualization_component.fs_label.setText(
            'Sampling rate = {0}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.stream_widget_visualization_component.ts_label.setText(
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
        except ZeroDivisionError:
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
        self.viz_time_vector = np.linspace(0., get_stream_preset_info(self.stream_name, 'DisplayDuration'),
                                           num_points_to_plot)

    def update_sr_and_display_duration_in_settings(self, new_sampling_rate, new_display_duration):
        '''
        this function is called by StreamWidget when on_num_points_to_display_change is called
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        set_stream_preset_info(self.stream_name, 'NominalSamplingRate', new_sampling_rate)
        set_stream_preset_info(self.stream_name, 'DisplayDuration', new_display_duration)

    def reload_visualization_elements(self, info_dict):
        self.group_info = collect_stream_all_groups_info(self.stream_name)
        clear_layout(self.MetaInfoVerticalLayout)
        clear_layout(self.TimeSeriesPlotsLayout)
        clear_layout(self.ImageWidgetLayout)
        clear_layout(self.BarPlotWidgetLayout)
        self.create_visualization_component()

    def plot_format_on_change(self, info_dict):
        old_format = self.group_info[info_dict['group_name']]['selected_plot_format']
        self.preset_on_change()

        self.stream_widget_visualization_component.plot_elements[plot_format_index_dict[old_format]][
            info_dict['group_name']].hide()
        self.stream_widget_visualization_component.plot_elements[plot_format_index_dict[info_dict['new_format']]][
            info_dict['group_name']].show()

        # update the plot hide display

    def preset_on_change(self):
        self.group_info = collect_stream_all_groups_info(self.stream_name)

    def get_image_format_and_shape(self, group_name):
        width = self.group_info[group_name]['plot_format']['image']['width']
        height = self.group_info[group_name]['plot_format']['image']['height']
        image_format = self.group_info[group_name]['plot_format']['image']['image_format']
        depth = image_depth_dict[image_format]
        channel_format = self.group_info[group_name]['plot_format']['image']['channel_format']
        scaling_factor = self.group_info[group_name]['plot_format']['image']['scaling_factor']

        return width, height, depth, image_format, channel_format, scaling_factor

    #############################################

    def bar_chart_range_on_change(self, stream_name, group_name):
        self.preset_on_change()
        if not self.group_info[group_name]['is_image_only']:  # if barplot exists for this group
            widget = self.stream_widget_visualization_component.plot_elements['bar_chart'][group_name]
            widget.setYRange(min=self.group_info[group_name]['plot_format']['bar_chart']['y_min'],
                             max=self.group_info[group_name]['plot_format']['bar_chart']['y_max'])

    def set_plot_widget_range(self, x_min, x_max, y_min, y_max):

        return

#############################################
