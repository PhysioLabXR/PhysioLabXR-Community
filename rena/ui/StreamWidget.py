# This Python file uses the following encoding: utf-8

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QLabel
from pyqtgraph import PlotDataItem

from rena import config, config_ui
from rena.threadings import workers
from rena.ui.OptionsWindow import OptionsWindow
from rena.ui.StreamWidgetVisualizationComponents import StreamWidgetVisualizationComponents
from rena.ui_shared import start_stream_icon, stop_stream_icon, pop_window_icon, dock_window_icon, remove_stream_icon, \
    options_icon
from rena.utils.ui_utils import AnotherWindow, dialog_popup, get_distinct_colors


class StreamWidget(QtWidgets.QWidget):
    def __init__(self, main_parent, parent, stream_name, preset, interface, insert_position=None):
        """
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
        self.stream_name = stream_name
        self.preset = preset

        self.StreamNameLabel.setText(stream_name)
        self.set_button_icons()
        self.OptionsBtn.setIcon(options_icon)
        self.RemoveStreamBtn.setIcon(remove_stream_icon)

        # visualization data buffer
        self.current_timestamp = 0
        self.lsl_data_buffer = np.empty(shape=(self.preset['NumChannels'], 0))

        # timer
        self.timer = QTimer()
        self.timer.setInterval(config.REFRESH_INTERVAL)  # for 1000 Hz refresh rate
        self.timer.timeout.connect(self.ticks)
        self.timer.start()

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(config.VISUALIZATION_REFRESH_INTERVAL)  # for 15 Hz refresh rate
        self.v_timer.timeout.connect(self.visualize_LSLStream_data)
        self.v_timer.start()

        # connect btn
        self.StartStopStreamBtn.clicked.connect(self.start_stop_stream_btn_clicked)
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        # visualization component
        self.stream_widget_visualization_component = None

        # data elements
        self.worker_thread = pg.QtCore.QThread(self)
        self.interface = interface
        self.lsl_worker = workers.LSLInletWorker(interface)
        self.lsl_worker.signal_data.connect(self.process_LSLStream_data)
        self.lsl_worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        # create visualization component:
        self.create_visualization_component()

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
        signal_settings_window = OptionsWindow(parent=self, preset=self.preset)
        if signal_settings_window.exec_():
            print("signal setting window open")
        else:
            print("Cancel!")

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.lsl_worker.is_streaming:
            self.lsl_worker.stop_stream()
            if not self.lsl_worker.is_streaming:
                # started
                print("sensor stopped")
                # toggle the icon
                self.StartStopStreamBtn.setText("Start Stream")
        else:
            self.lsl_worker.start_stream()
            if self.lsl_worker.is_streaming:
                # started
                print("sensor stopped")
                # toggle the icon
                self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()

    def dock_window(self):
        self.parent.insertWidget(self.parent.count() - 1,
                                 self)
        self.PopWindowBtn.clicked.disconnect()
        self.PopWindowBtn.clicked.connect(self.pop_window)
        self.PopWindowBtn.setText('Pop Window')
        self.main_parent.pop_windows[self.stream_name].hide()  # tetentive measures
        self.main_parent.pop_windows.pop(self.stream_name)
        self.set_button_icons()

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
        self.worker_thread.exit()
        # self.main_parent.lsl_workers.pop(self.stream_name)
        # self.main_parent.worker_threads.pop(self.stream_name)
        # if this lsl connect to a device:

        # TODO: we do not consider device at this stage
        # if self.stream_name in self.main_parent.device_workers.keys():
        #     self.main_parent.device_workers[self.stream_name].stop_stream()
        #     self.main_parent.device_workers.pop(self.stream_name)

        self.main_parent.stream_widgets.pop(self.stream_name)
        self.parent.removeWidget(self)
        # close window if popped
        if self.stream_name in self.main_parent.pop_windows.keys():
            self.main_parent.pop_windows[self.stream_name].hide()
            # self.main_parent.pop_windows.pop(self.stream_name)
            self.deleteLater()
        else:  # use recursive delete if docked
            self.deleteLater()
        # self.main_parent.LSL_data_buffer_dicts.pop(self.stream_name)
        return True

    def init_visualize_LSLStream_data(self, parent, metainfo_parent, preset):

        # init stream view with LSL
        # lsl_num_chan, channel_names, group_channels_in_plot = preset['NumChannels'], \
        #                                                                     preset['ChannelNames'], \
        #                                                                     preset['GroupChannelsInPlot']

        fs_label = QLabel(text='Sampling rate = ')
        ts_label = QLabel(text='Current Time Stamp = ')
        metainfo_parent.addWidget(fs_label)
        metainfo_parent.addWidget(ts_label)
        # if plot_group_slices:
        plot_widgets = {}
        plots = []
        # plot_formats = []
        for group_name in preset['GroupChannelsInPlot']:
            plot_group_info = preset['GroupChannelsInPlot'][group_name]
            plot_format = plot_group_info['plot_format']
            # one plot widget for each group, no need to check chan_names because plot_group_slices only comes with preset
            if plot_format == 'time_series':  # time_series plot
                # plot_formats.append(plot_group_format_info)
                plot_widget = pg.PlotWidget()
                parent.addWidget(plot_widget)

                distinct_colors = get_distinct_colors(len(plot_group_info['channels']))
                plot_widget.addLegend()

                plot_data_items = []
                for channel_index_in_group, channel_name in enumerate([preset['ChannelNames'][i] for i in plot_group_info['channels']]):
                    if plot_group_info['channels_display'][channel_index_in_group]: # if display is 1
                        plot_data_items.append(plot_widget.plot([], [], pen=pg.mkPen(color=distinct_colors[channel_index_in_group]), name=channel_name))
                    else:
                        plot_data_items.append(None)

                plots.append(plot_data_items)

                # for plot_index, (color, c_name) in enumerate(distinct_colors, [preset['ChannelNames'][i] for i in plot_group_info['channels']]):
                #     print("John")
                #
                # plots.append([plot_widget.plot([], [], pen=pg.mkPen(color=color), name=c_name) for color, c_name in
                #               zip(distinct_colors, [preset['ChannelNames'][i] for i in plot_group_info['channels']])])

                if plot_group_info['group_display'] == 0:
                    plot_widget.hide()
                plot_widgets[group_name] = plot_widget

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

        return fs_label, ts_label, plot_widgets, plots, preset

    def create_visualization_component(self):
        # TODO: try catch group format error
        fs_label, ts_label, plot_widgets, plots, preset = \
            self.init_visualize_LSLStream_data(parent=self.TimeSeriesPlotsLayout,
                                               metainfo_parent=self.MetaInfoVerticalLayout,
                                               preset=self.preset)
        self.stream_widget_visualization_component = \
            StreamWidgetVisualizationComponents(fs_label, ts_label, plot_widgets, plots, preset)

    def process_LSLStream_data(self, data_dict):
        samples_to_plot = self.preset["num_samples_to_plot"]
        if data_dict['frames'].shape[-1] > 0:
            buffered_data = self.lsl_data_buffer
            try:
                buffered_data = np.concatenate(
                    (buffered_data, data_dict['frames']),
                    axis=-1)  # get all data and remove it from internal buffer
                self.current_timestamp = data_dict['timestamps'][-1]
            except ValueError:
                raise Exception('The number of channels for stream {0} mismatch from its preset json.'.format(
                    data_dict['lsl_data_type']))
            if buffered_data.shape[-1] < samples_to_plot:
                data_to_plot = np.concatenate((np.zeros(shape=(
                    buffered_data.shape[0],
                    samples_to_plot -
                    buffered_data.shape[-1])),
                                               buffered_data), axis=-1)
            else:
                data_to_plot = buffered_data[:,
                               - samples_to_plot:]  # plot the most recent few seconds

            # main window only retains the most recent 10 seconds for visualization purposes
            self.lsl_data_buffer = data_to_plot
            self.preset["ActualSamplingRate"] = data_dict['sampling_rate']
            # notify the internal buffer in recordings tab

            # reshape data_dict based on sensor interface
            self.main_parent.recording_tab.update_buffers(data_dict)

            # inference tab
            # self.main_parent.inference_tab.update_buffers(data_dict)

    def visualize_LSLStream_data(self):

        # for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
        time_vector = self.preset["timevector"]
        data_to_plot = self.lsl_data_buffer
        if data_to_plot.shape[-1] == len(time_vector):
            actual_sampling_rate = self.preset["ActualSamplingRate"]
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
            for plot_group_index, (group_name) in enumerate(self.preset["GroupChannelsInPlot"]):
                plot_group_info = self.preset["GroupChannelsInPlot"][group_name]
                if plot_group_info["plot_format"] == 'time_series':
                    # plot corresponding time series data, range (a,b) is time series
                    # plot_group_channel_num = len(plot_group_info['channels'])
                    for index_in_group, channel_index in enumerate(plot_group_info['channels']):
                        if plot_group_info['channels_display'][index_in_group]:
                            # print(channel_index)
                            self.stream_widget_visualization_component.plots[plot_group_index][index_in_group] \
                                .setData(time_vector, data_to_plot[channel_index, :])
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
        self.lsl_worker.tick_signal.emit()
        #     self.recent_tick_refresh_timestamps.append(time.time())
        #     if len(self.recent_tick_refresh_timestamps) > 2:
        #         self.tick_rate = 1 / ((self.recent_tick_refresh_timestamps[-1] - self.recent_tick_refresh_timestamps[0]) / (
        #                     len(self.recent_tick_refresh_timestamps) - 1))
        #
        #     self.tickFrequencyLabel.setText(
        #         'Pull Data Frequency: {0}'.format(round(self.tick_rate, config_ui.tick_frequency_decimal_places)))
