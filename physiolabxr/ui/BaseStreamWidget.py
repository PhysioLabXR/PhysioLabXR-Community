# This Python file uses the following encoding: utf-8
import time
from collections import deque
from typing import Callable

from PyQt6 import QtWidgets, uic, QtCore
from PyQt6.QtCore import QTimer, QThread, QMutex, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialogButtonBox, QSplitter

from physiolabxr.configs import config_ui
from physiolabxr.configs.configs import AppConfigs, LinechartVizMode
from physiolabxr.presets.load_user_preset import create_default_group_entry
from physiolabxr.presets.presets_utils import get_stream_preset_info, set_stream_preset_info, get_stream_group_info, \
    get_is_group_shown, pop_group_from_stream_preset, add_group_entry_to_stream, change_stream_group_order, \
    change_stream_group_name, pop_stream_preset_from_settings, change_group_channels, reset_all_group_data_processors, \
    get_stream_data_processor_only_apply_to_visualization
from physiolabxr.ui.GroupPlotWidget import GroupPlotWidget
from physiolabxr.ui.PoppableWidget import Poppable
from physiolabxr.ui.StreamOptionsWindow import StreamOptionsWindow
from physiolabxr.ui.VizComponents import VizComponents
from physiolabxr.utils.buffers import DataBufferSingleStream
from physiolabxr.utils.dsp_utils.dsp_modules import run_data_processors
from physiolabxr.utils.performance_utils import timeit
from physiolabxr.utils.ui_utils import dialog_popup, clear_widget


class BaseStreamWidget(Poppable, QtWidgets.QWidget):
    plot_format_changed_signal = QtCore.pyqtSignal(dict)
    channel_mismatch_buttons = buttons=QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No

    def __init__(self, parent_widget, parent_layout, preset_type, stream_name, data_timer_interval, use_viz_buffer, insert_position=None, option_widget_call: callable=None):
        """
        This is the base stream widget class. It contains the main interface for a single stream.

        pull_data_timer and viz v_timer are initialized and started here. Class extending this class will be responsible
        for starting the timers.

        @param parent_widget: the MainWindow class
        @param parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
        @param stream_name: the name of the stream
        @param use_viz_buffer: whether to use a buffer for visualization. If set to false, the child class must override
        the visualize function. Video stream including webcam and screen capture does not use viz buffer
        """
        super().__init__(stream_name, parent_widget, parent_layout, self.remove_stream)
        self.ui = uic.loadUi(AppConfigs()._ui_StreamWidget, self)
        self.set_pop_button(self.PopWindowBtn)
        self.StreamNameLabel.setText(stream_name)
        self.OptionsBtn.setIcon(AppConfigs()._icon_options)
        self.RemoveStreamBtn.setIcon(AppConfigs()._icon_remove)

        if type(insert_position) == int:
            parent_layout.insertWidget(insert_position, self)
        else:
            parent_layout.addWidget(self)
        self.parent = parent_layout
        self.main_parent = parent_widget
        # add splitter to the layout
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.viz_group_scroll_layout.addWidget(self.splitter)

        self.preset_type = preset_type
        self.stream_name = stream_name  # this also keeps the subtopic name if using ZMQ
        self.actualSamplingRate = 0
        self.current_timestamp = 0
        self.is_stream_available = False
        self.in_error_state = False  # an error state to prevent ticking when is set to true

        # visualization timer
        self.data_timer = QTimer()
        self.data_timer.setInterval(int(float(data_timer_interval)))
        self.data_timer.timeout.connect(self.pull_data_tick)
        self.v_timer = QTimer()
        self.v_timer.setInterval(int(float(AppConfigs().visualization_refresh_interval)))
        self.v_timer.timeout.connect(self.visualize)

        # connect btn
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        # inefficient loading of assets TODO need to confirm creating Pixmap in ui_shared result in crash
        self.stream_unavailable_pixmap = QPixmap(AppConfigs()._stream_unavailable)
        self.stream_available_pixmap = QPixmap(AppConfigs()._stream_available)
        self.stream_active_pixmap = QPixmap(AppConfigs()._stream_viz_active)

        # visualization components
        self.viz_components = None  # stores all the visualization components we initialize it in the init_stream_visualization()
        self.num_points_to_plot = None

        # data elements
        if use_viz_buffer:
            self.viz_data_buffer = None
            self.create_buffer()

        # create visualization component, must be after the option window ##################
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        self.create_visualization_component()

        self._has_new_viz_data = False
        self.viz_data_head = 0

        # create option window
        if option_widget_call is None:
            self.option_window = StreamOptionsWindow(parent_stream_widget=self, stream_name=self.stream_name, plot_format_changed_signal=self.plot_format_changed_signal)
        else:
            self.option_window = option_widget_call()
        self.option_window.hide()

        # FPS counter``
        self.viz_times = None
        self.update_buffer_times = None
        self.plot_data_times = None
        self.reset_performance_measures()

        # mutex for not update the settings while plotting
        self.setting_update_viz_mutex = QMutex()
        self.set_pop_button_icons()

    def start_timers(self):
        self.v_timer.start()
        self.data_timer.start()

    def connect_worker(self, worker, add_stream_availibility: bool):
        self.worker_thread = QThread(self)
        self.data_worker = worker
        self.add_stream_availability = add_stream_availibility
        self.data_worker.signal_data.connect(self.process_stream_data)
        if add_stream_availibility:
            self.data_worker.signal_stream_availability.connect(self.update_stream_availability)
        else:
            self.is_stream_available = True  # always true for stream that does not have stream availability
        self.data_worker.moveToThread(self.worker_thread)
        self.worker_thread.start()
        self.set_start_stop_button_icon()

    def connect_start_stop_btn(self, start_stop_callback: Callable):
        self.StartStopStreamBtn.clicked.connect(start_stop_callback)

    def start_stop_stream_btn_clicked(self):
        if self.data_worker.is_streaming:
            self.data_worker.stop_stream()
            if not self.data_worker.is_streaming and self.add_stream_availability:
                self.update_stream_availability(self.data_worker.is_stream_available)
        else:
            self.data_worker.start_stream()
        self.set_button_icons()
        self.main_parent.update_active_streams()

    def reset_performance_measures(self):
        self.update_buffer_times = []
        self.plot_data_times = []
        self.viz_times = deque(maxlen=10 * int(AppConfigs().visualization_refresh_interval))

    def update_stream_availability(self, is_stream_available):
        '''
        this function check if the stream is available
        '''
        print('Stream {0} availability is {1}'.format(self.stream_name, is_stream_available), end='\r')
        self.is_stream_available = is_stream_available
        if self.data_worker.is_streaming:
            if is_stream_available:
                if not self.StartStopStreamBtn.isEnabled(): self.StartStopStreamBtn.setEnabled(True)
                self.StreamAvailablilityLabel.setPixmap(self.stream_active_pixmap)
                self.StreamAvailablilityLabel.setToolTip("Stream {0} is being plotted".format(self.stream_name))
            else:
                self.start_stop_stream_btn_clicked()  # must stop the stream before dialog popup
                self.set_stream_unavailable()
                self.main_parent.current_dialog = dialog_popup('Lost connection to {0}'.format(self.stream_name), title='Warning', mode='modeless')
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
        self.set_pop_button_icons()
        self.set_start_stop_button_icon()

    def set_pop_button_icons(self):

        if not self.is_popped:
            self.PopWindowBtn.setIcon(AppConfigs()._icon_pop_window)
        else:
            self.PopWindowBtn.setIcon(AppConfigs()._icon_dock_window)

    def set_start_stop_button_icon(self):
        if not self.is_streaming():
            self.StartStopStreamBtn.setIcon(AppConfigs()._icon_start)
        else:
            self.StartStopStreamBtn.setIcon(AppConfigs()._icon_stop)

    def options_btn_clicked(self):
        print("Option window button clicked")
        self.option_window.show()
        self.option_window.activateWindow()

    def group_plot_widget_edit_option_clicked(self, group_name: str):
        self.options_btn_clicked()
        self.option_window.set_selected_group(group_name)

    def is_streaming(self):
        return self.data_worker.is_streaming

    def reset_preset_by_num_channels(self, num_channels, data_type, **kwargs):
        pop_stream_preset_from_settings(self.stream_name)
        self.main_parent.create_preset(self.stream_name, self.preset_type, data_type=data_type, num_channels=num_channels, **kwargs)  # update preset in settings
        self.create_buffer()  # recreate the interface and buffer, using the new preset
        self.data_worker.reset_interface(self.stream_name, get_stream_preset_info(self.stream_name, 'num_channels'))

        self.option_window.reload_preset_to_UI()
        self.reset_viz()

    def reset_viz(self):
        """
        caller to this function must ensure self.group_info is modified and up to date with user changes
        """
        self.clear_stream_visualizations()
        self.create_visualization_component()

    def create_buffer(self):
        num_channels = get_stream_preset_info(self.stream_name, 'num_channels')
        sr = get_stream_preset_info(self.stream_name, 'nominal_sampling_rate')
        display_duration = get_stream_preset_info(self.stream_name, 'display_duration')
        buffer_size = 1 if num_channels > AppConfigs.max_timeseries_num_channels_per_group else int(sr * display_duration)
        self.viz_data_buffer = DataBufferSingleStream(num_channels=num_channels, buffer_sizes=buffer_size, append_zeros=True)

    def remove_stream(self):

        if self.main_parent.recording_tab.is_recording:
            self.main_parent.current_dialog = dialog_popup(msg='Cannot remove stream while recording.')
            return False
        self.data_timer.stop()
        self.v_timer.stop()
        if self.data_worker.is_streaming:
            self.data_worker.stop_stream()
        self.worker_thread.requestInterruption()
        self.worker_thread.exit()
        self.worker_thread.wait()  # wait for the thread to exit

        self.main_parent.stream_widgets.pop(self.stream_name)
        self.main_parent.remove_stream_widget(self)
        # close window if popped
        if self.is_popped:
            self.delete_window()
        self.deleteLater()
        self.option_window.close()
        return True

    def update_channel_shown(self, channel_index, is_shown, group_name):
        channel_plot_widget = self.channel_index_plot_widget_dict[channel_index]
        channel_plot_widget.show() if is_shown else channel_plot_widget.hide()
        self.update_groups_shown(group_name)

    def update_groups_shown(self, group_name):
        # assuming group info is update to date with in the persist settings
        # check if there's active channels in this group
        if get_is_group_shown(self.stream_name, group_name):
            self.group_name_plot_widget_dict[group_name].show()
        else:
            self.group_name_plot_widget_dict[group_name].hide()

    def clear_stream_visualizations(self):
        self.channel_index_plot_widget_dict = {}
        self.group_name_plot_widget_dict = {}
        clear_widget(self.splitter)

    def init_stream_visualization(self):
        group_plot_widget_dict = {}
        channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
        group_info = get_stream_group_info(self.stream_name)
        for group_name in group_info.keys():
            group_channel_names = None if channel_names is None else [channel_names[int(i)] for i in group_info[group_name].channel_indices]
            group_plot_widget_dict[group_name] = GroupPlotWidget(self, self.stream_name, group_name, group_channel_names, get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'), self.plot_format_changed_signal)
            self.splitter.addWidget(group_plot_widget_dict[group_name])
            self.num_points_to_plot = self.get_num_points_to_plot()

        return group_plot_widget_dict

    def create_visualization_component(self):
        group_plot_dict = self.init_stream_visualization()
        self.viz_components = VizComponents(self.fs_label, self.ts_label, group_plot_dict)

    def process_stream_data(self, data_dict):
        '''
        update the visualization buffer, recording buffer, and scripting buffer
        '''
        if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
            # if only applied to visualization, then only update the visualization buffer
            if get_stream_data_processor_only_apply_to_visualization(self.stream_name):
                self.main_parent.recording_tab.update_recording_buffer(data_dict)
                self.main_parent.scripting_tab.forward_data(data_dict)
                self.run_data_processor(data_dict) # run data processor after updating recording buffer and scripting buffer
                self.viz_data_head = self.viz_data_head + len(data_dict['timestamps'])
            else:
                # run data processor first
                self.run_data_processor(data_dict)
                self.main_parent.recording_tab.update_recording_buffer(data_dict)
                self.main_parent.scripting_tab.forward_data(data_dict)
                self.viz_data_head = self.viz_data_head + len(data_dict['timestamps'])


            # self.run_data_processor(data_dict)
            # self.viz_data_head = self.viz_data_head + len(data_dict['timestamps'])
            self.update_buffer_times.append(timeit(self.viz_data_buffer.update_buffer, (data_dict, ))[1])  # NOTE performance test scripts, don't include in production code
            self._has_new_viz_data = True

            self.actualSamplingRate = data_dict['sampling_rate']
            self.current_timestamp = data_dict['timestamps'][-1]



            # if data_dict['frames'].shape[
            #     -1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
            #     self.run_data_processor(data_dict)
            #     self.viz_data_head = self.viz_data_head + len(data_dict['timestamps'])
            #     self.update_buffer_times.append(timeit(self.viz_data_buffer.update_buffer, (data_dict,))[
            #                                         1])  # NOTE performance test scripts, don't include in production code
            #     self._has_new_viz_data = True
            #
            #     self.actualSamplingRate = data_dict['sampling_rate']
            #     self.current_timestamp = data_dict['timestamps'][-1]
            #     # notify the internal buffer in recordings tab
            #
            #     # reshape data_dict based on sensor interface
            #     self.main_parent.recording_tab.update_recording_buffer(data_dict)
            #     self.main_parent.scripting_tab.forward_data(data_dict)
            #     # scripting tab

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

        self.viz_times.append(time.time())
        self.data_worker.signal_stream_availability_tick.emit()  # signal updating the stream availability
        actual_sampling_rate = self.actualSamplingRate
        if not self._has_new_viz_data:
            return

        if AppConfigs().linechart_viz_mode == LinechartVizMode.INPLACE:
            data_to_plot = self.viz_data_buffer.buffer[0][:, -self.viz_data_head:]
        elif AppConfigs().linechart_viz_mode == LinechartVizMode.CONTINUOUS:
            data_to_plot = self.viz_data_buffer.buffer[0][:, -self.num_points_to_plot:]
        for plot_group_index, (group_name) in enumerate(get_stream_group_info(self.stream_name).keys()):
            self.plot_data_times.append(timeit(self.viz_components.group_plots[group_name].plot_data, (data_to_plot, ))[1])  # NOTE performance test scripts, don't include in production code

        self.viz_components.fs_label.setText(
            'fps: {:.3f}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.viz_components.ts_label.setText('timestamp: {:.3f}'.format(self.current_timestamp))

        self._has_new_viz_data = False
        if self.viz_data_head > get_stream_preset_info(self.stream_name, 'display_duration') * get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'):  # reset the head if it is out of bound
            self.viz_data_head = 0

    def pull_data_tick(self):
        self.data_worker.signal_data_tick.emit()

    def get_fps(self):
        try:
            return len(self.viz_times) / (self.viz_times[-1] - self.viz_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0

    def is_widget_streaming(self):
        return self.data_worker.is_streaming

    def on_num_points_to_display_change(self):
        '''
        this function is called by StreamOptionWindow when user change number of points to plot by changing
        the sampling rate or the display duration.
        Changing the num points to plot here will cause, in the next plot cycle, GroupPlotWidget will have a mismatch between
        the data received from here and its time vector. This will cause the GroupPlotWidget to update its time vector
        :param new_sampling_rate:
        :param new_display_duration:
        :return:
        '''
        self.create_buffer()
        self.num_points_to_plot = self.get_num_points_to_plot()
        if self.viz_components is not None:
            self.viz_components.update_nominal_sampling_rate()

    def bar_chart_range_on_change(self, group_name):
        self.viz_components.group_plots[group_name].update_bar_chart_range()


    def channel_group_changed(self, change_dict):
        """
        Called when one or more channel's parent group is changed
        @param change_dict:
        """
        # update the group info
        for group_name, child_channels in change_dict.items():
            if len(child_channels) == 0:
                pop_group_from_stream_preset(self.stream_name, group_name)
            else:  # cover the cases for both changed groups and new group
                channel_indices = [x.lsl_index for x in child_channels]
                is_channels_shown = [x.is_shown for x in child_channels]
                if group_name not in get_stream_group_info(self.stream_name).keys():  # if this is a new group
                    add_group_entry_to_stream(self.stream_name, create_default_group_entry(len(child_channels), group_name, channel_indices=channel_indices, is_channels_shown=is_channels_shown))
                else:
                    change_group_channels(self.stream_name, group_name, channel_indices, is_channels_shown)

        # reset data processor
        # TODO: optimize for changed group reset. Reset visualization buffer after regrouped ?
        reset_all_group_data_processors(self.stream_name)

        # save_preset()
        self.reset_viz()

    def group_order_changed(self, group_order):
        """
        Called when the group order is changed
        @param group_order:
        """
        change_stream_group_order(self.stream_name, group_order)
        # save_preset()
        self.reset_viz()

    def change_group_name(self, new_group_name, old_group_name):
        try:
            change_stream_group_name(self.stream_name, new_group_name, old_group_name)
        except ValueError as e:
            dialog_popup(str(e), mode='modeless')
        self.viz_components.group_plots[new_group_name] = self.viz_components.group_plots.pop(old_group_name)
        self.viz_components.group_plots[new_group_name].change_group_name(new_group_name)

    def change_channel_name(self, group_name, new_ch_name, old_ch_name, lsl_index):
        # change channel name in the settings
        channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
        changing_channel_index = channel_names.index(old_ch_name)
        channel_names[changing_channel_index] = new_ch_name
        set_stream_preset_info(self.stream_name, 'channel_names', channel_names)

        # change the name in the plots
        self.viz_components.group_plots[group_name].change_channel_name(new_ch_name, old_ch_name, lsl_index)

    def get_num_points_to_plot(self):
        display_duration = get_stream_preset_info(self.stream_name, 'display_duration')
        return int(display_duration * get_stream_preset_info(self.stream_name, 'nominal_sampling_rate'))

    def get_pull_data_delay(self):
        return self.data_worker.get_pull_data_delay()

    def set_spectrogram_cmap(self, group_name):
        self.viz_components.set_spectrogram_cmap(group_name)

    def try_close(self):
        return self.remove_stream()

    def run_data_processor(self, data_dict):
        data = data_dict['frames']
        group_info = get_stream_group_info(self.stream_name)

        for this_group_info in group_info.values():  # TODO: potentially optimize using pool
            if len(this_group_info.data_processors) != 0:
                processed_data = run_data_processors(data[this_group_info.channel_indices], this_group_info.data_processors)
                data[this_group_info.channel_indices] = processed_data


    def get_viz_components(self):
        return self.viz_components