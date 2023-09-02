# This Python file uses the following encoding: utf-8
from collections import deque

from PyQt6 import QtCore
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QMutex
from PyQt6.QtCore import QTimer, QThread
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QDialogButtonBox

from physiolabxr.configs import config
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.examples.fmri_experiment_example.mri_utils import *
from physiolabxr.exceptions.exceptions import ChannelMismatchError, LSLStreamNotFoundError, UnsupportedErrorTypeError
from physiolabxr.presets.presets_utils import get_stream_preset_info, pop_stream_preset_from_settings, \
    reset_all_group_data_processors
from physiolabxr.threadings import workers
# get_mri_coronal_view_dimension, get_mri_sagittal_view_dimension, \
#     get_mri_axial_view_dimension
from physiolabxr.ui.PoppableWidget import Poppable
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel
from physiolabxr.utils.buffers import DataBufferSingleStream
from physiolabxr.utils.ui_utils import dialog_popup


# This Python file uses the following encoding: utf-8

class FMRIWidget(Poppable, QtWidgets.QWidget):
    plot_format_changed_signal = QtCore.pyqtSignal(dict)
    channel_mismatch_buttons = buttons = QDialogButtonBox.Yes | QDialogButtonBox.No

    def __init__(self, parent_widget, parent_layout, stream_name, data_type, worker, networking_interface, port_number,
                 insert_position=None, ):
        """
        StreamWidget is the main interface with plots and a single stream of data.
        The stream can be either LSL or ZMQ.
        @param parent_widget: the MainWindow
        @param parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
        """

        # GUI elements
        super().__init__(stream_name, parent_widget, parent_layout, self.remove_stream)
        self.ui = uic.loadUi("examples/fmri_experiment_example/FMRIWidgetNew._ui", self)
        self.set_pop_button(self.PopWindowBtn)

        # if type(insert_position) == int:
        #     parent_layout.insertWidget(insert_position, self)
        # else:
        #     parent_layout.addWidget(self)

        self.parent = parent_layout
        self.main_parent = parent_widget

        ##
        self.stream_name = stream_name  # this also keeps the subtopic name if using ZMQ
        self.networking_interface = networking_interface
        self.port_number = port_number
        self.data_type = data_type
        # self.preset = get_complete_stream_preset_info(self.stream_name)

        self.actualSamplingRate = 0

        self.StreamNameLabel.setText(stream_name)
        self.OptionsBtn.setIcon(AppConfigs()._icon_options)
        self.RemoveStreamBtn.setIcon(AppConfigs()._icon_remove_stream)

        self.is_stream_available = False
        self.in_error_state = False  # an error state to prevent ticking when is set to true

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
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        # inefficient loading of assets TODO need to confirm creating Pixmap in ui_shared result in crash
        self.stream_unavailable_pixmap = QPixmap(AppConfigs()._stream_unavailable)
        self.stream_available_pixmap = QPixmap(AppConfigs()._stream_available)
        self.stream_active_pixmap = QPixmap(AppConfigs()._stream_viz_active)

        # visualization component
        # This variable stores all the visualization components we initialize it in the init_stream_visualization()

        #self.viz_components = None
        #self.num_points_to_plot = None

        # data elements
        self.viz_data_buffer = None
        self.create_buffer()

        self.worker_thread = QThread(self)
        self.worker = workers.ZMQWorker(port_number=port_number, subtopic=stream_name, data_type=data_type)
        self.worker.signal_data.connect(self.process_stream_data)
        self.worker.signal_stream_availability.connect(self.update_stream_availability)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

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
        self.update_buffer_times = []
        self.plot_data_times = []
        ########################################################################

        self.init_mri_graphic_components()
        self.init_fmri_graphic_component()

        self.load_mri_volume()
        # self.load_fmri_volume()

    def __post_init__(self):
        self.sagittal_view_slider_value = 0
        self.coronal_view_slider_value = 0
        self.axial_view_slider_value = 0
        # self.fmri_timestamp_slider_value = 0

    def init_mri_graphic_components(self):
        ######################################################
        self.volume_view_plot = gl.GLViewWidget()
        self.VolumnViewPlotWidget.layout().addWidget(self.volume_view_plot)

        ######################################################
        self.sagittal_view_plot = pg.PlotWidget()
        self.SagittalViewPlotWidget.layout().addWidget(self.sagittal_view_plot)
        self.sagittal_view_mri_image_item = pg.ImageItem()
        self.sagittal_view_fmri_image_item = pg.ImageItem()
        self.sagittal_view_plot.addItem(self.sagittal_view_mri_image_item)
        self.sagittal_view_plot.addItem(self.sagittal_view_fmri_image_item)

        self.sagittal_view_slider = SliderWithValueLabel()
        self.sagittal_view_slider.valueChanged.connect(self.sagittal_view_slider_on_change)
        self.SagittalViewSliderWidget.layout().addWidget(self.sagittal_view_slider)

        ######################################################
        self.coronal_view_plot = pg.PlotWidget()
        self.CoronalViewPlotWidget.layout().addWidget(self.coronal_view_plot)
        self.coronal_view_mri_image_item = pg.ImageItem()
        self.coronal_view_fmri_image_item = pg.ImageItem()
        self.coronal_view_plot.addItem(self.coronal_view_mri_image_item)
        self.coronal_view_plot.addItem(self.coronal_view_fmri_image_item)

        self.coronal_view_slider = SliderWithValueLabel()
        self.coronal_view_slider.valueChanged.connect(self.coronal_view_slider_on_change)
        self.CoronalViewSliderWidget.layout().addWidget(self.coronal_view_slider)

        ######################################################
        self.axial_view_plot = pg.PlotWidget()
        self.AxiaViewPlotWidget.layout().addWidget(self.axial_view_plot)
        self.axial_view_mri_image_item = pg.ImageItem()
        self.axial_view_fmri_image_item = pg.ImageItem()
        self.axial_view_plot.addItem(self.axial_view_mri_image_item)
        self.axial_view_plot.addItem(self.axial_view_fmri_image_item)

        self.axial_view_slider = SliderWithValueLabel()
        self.axial_view_slider.valueChanged.connect(self.axial_view_slider_on_change)
        self.AxiaViewSliderWidget.layout().addWidget(self.axial_view_slider)

    def init_fmri_graphic_component(self):
        self.fmri_timestamp_slider = SliderWithValueLabel()
        # self.fmri_timestamp_slider.valueChanged.connect(self.fmri_timestamp_slider_on_change)
        self.FMRITimestampSliderWidget.layout().addWidget(self.fmri_timestamp_slider)

    def load_mri_volume(self):
        # g = gl.GLGridItem()
        # g.scale(100, 100, 100)
        # self.volume_view_plot.addItem(g)

        _, self.mri_volume_data = load_nii_gz_file(
            'C:/Users/Haowe/OneDrive/Desktop/Columbia/RENA/RealityNavigation/physiolabxr/examples/fmri_experiment_example/structural.nii.gz')
        self.gl_volume_item = volume_to_gl_volume_item(self.mri_volume_data)
        self.volume_view_plot.addItem(self.gl_volume_item)
        self.set_mri_view_slider_range()

    # def load_fmri_volume(self):
    #     _, self.fmri_volume_data = load_nii_gz_file(
    #         'C:/Users/Haowe/OneDrive/Desktop/Columbia/RENA/RealityNavigation/physiolabxr/examples/fmri_experiment_example/resampled_fmri.nii.gz')
    #     self.set_fmri_view_slider_range()

    def set_mri_view_slider_range(self):
        # # coronal view, sagittal view, axial view
        # x_size, y_size, z_size = self.volume_data.shape
        self.coronal_view_slider.setRange(minValue=0, maxValue=get_mri_coronal_view_dimension(self.mri_volume_data) - 1)
        self.sagittal_view_slider.setRange(minValue=0,
                                           maxValue=get_mri_sagittal_view_dimension(self.mri_volume_data) - 1)
        self.axial_view_slider.setRange(minValue=0, maxValue=get_mri_axial_view_dimension(self.mri_volume_data) - 1)

        self.coronal_view_slider.setValue(0)
        self.sagittal_view_slider.setValue(0)
        self.axial_view_slider.setValue(0)

    def set_fmri_view_slider_range(self):
        self.fmri_timestamp_slider.setRange(minValue=0,
                                            maxValue=self.fmri_volume_data.shape[-1] - 1)

        self.fmri_timestamp_slider.setValue(0)

    def coronal_view_slider_on_change(self):
        self.coronal_view_slider_value = self.coronal_view_slider.value()
        self.coronal_view_mri_image_item.setImage(
            get_mri_coronal_view_slice(self.mri_volume_data, index=self.coronal_view_slider_value))

        self.set_coronal_view_fmri()

    def sagittal_view_slider_on_change(self):
        self.sagittal_view_slider_value = self.sagittal_view_slider.value()
        self.sagittal_view_mri_image_item.setImage(
            get_mri_sagittal_view_slice(self.mri_volume_data, index=self.sagittal_view_slider_value))

        self.set_sagittal_view_fmri()

    def axial_view_slider_on_change(self):
        self.axial_view_slider_value = self.axial_view_slider.value()
        self.axial_view_mri_image_item.setImage(
            get_mri_axial_view_slice(self.mri_volume_data, index=self.axial_view_slider_value))

        self.set_axial_view_fmri()

    def set_coronal_view_fmri(self):
        pass
        # fmri_slice = get_fmri_coronal_view_slice(self.fmri_volume_data, self.coronal_view_slider_value,
        #                                          self.fmri_timestamp_slider_value)
        # self.coronal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))

    def set_sagittal_view_fmri(self):
        pass
        # fmri_slice = get_fmri_coronal_view_slice(self.fmri_volume_data, self.sagittal_view_slider_value,
        #                                          self.fmri_timestamp_slider_value)
        # self.sagittal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))

    def set_axial_view_fmri(self):
        pass
        # fmri_slice = get_fmri_axial_view_slice(self.fmri_volume_data, self.axial_view_slider_value,
        #                                        self.fmri_timestamp_slider_value)
        # self.axial_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))











    def reset_performance_measures(self):
        self.update_buffer_times = []
        self.plot_data_times = []
        self.tick_times = deque(maxlen=10 * int(float(config.settings.value('visualization_refresh_interval'))))

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
                self.main_parent.current_dialog = dialog_popup('Lost connection to {0}'.format(self.stream_name),
                                                               title='Warning', mode='modeless')
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
            self.StartStopStreamBtn.setIcon(AppConfigs()._icon_start)
        else:
            self.StartStopStreamBtn.setIcon(AppConfigs()._icon_stop)

        if not self.is_popped:
            self.PopWindowBtn.setIcon(AppConfigs()._icon_pop_window)
        else:
            self.PopWindowBtn.setIcon(AppConfigs()._icon_dock_window)

    def options_btn_clicked(self):
        pass
        # print("Option window button clicked")
        # self.stream_options_window.show()
        # self.stream_options_window.activateWindow()

    # def group_plot_widget_edit_option_clicked(self, group_name: str):
    #     self.options_btn_clicked()
    #     self.stream_options_window.set_selected_group(group_name)

    def is_streaming(self):
        return self.worker.is_streaming

    def start_stop_stream_btn_clicked(self):
        # check if is streaming
        if self.worker.is_streaming:
            self.worker.stop_stream()
            if not self.worker.is_streaming:
                # started
                # print("sensor stopped")
                # self.StartStopStreamBtn.setText("Start Stream")  # toggle the icon
                self.update_stream_availability(self.worker.is_stream_available)
        else:
            # self.reset_performance_measures()
            try:
                reset_all_group_data_processors(self.stream_name)
                self.worker.start_stream()
            except LSLStreamNotFoundError as e:
                self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
                return
            except ChannelMismatchError as e:  # only LSL's channel mismatch can be checked at this time, zmq's channel mismatch can only be checked when receiving data
                # self.main_parent.current_dialog = reply = QMessageBox.question(self, 'Channel Mismatch',
                #                              'The stream with name {0} found on the network has {1}.\n'
                #                              'The preset has {2} channels. \n '
                #                              'Do you want to reset your preset to a default and start stream.\n'
                #                              'You can edit your stream channels in Options if you choose No'.format(
                #                                  self.stream_name, e.message,
                #                                  len(get_stream_preset_info(self.stream_name, 'ChannelNames'))),
                #                              QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                preset_chan_num = len(get_stream_preset_info(self.stream_name, 'channel_names'))
                message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
                reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent,
                                     buttons=self.channel_mismatch_buttons)

                if reply.result():
                    self.reset_preset_by_num_channels(e.message)
                    try:
                        self.worker.start_stream()  # start the stream again with updated preset
                    except LSLStreamNotFoundError as e:
                        self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
                        return
                else:
                    return
            except Exception as e:
                raise UnsupportedErrorTypeError(str(e))
            # if self.worker.is_streaming:
            #     self.StartStopStreamBtn.setText("Stop Stream")
        self.set_button_icons()
        self.main_parent.update_active_streams()

    def reset_preset_by_num_channels(self, num_channels):
        pop_stream_preset_from_settings(self.stream_name)
        self.main_parent.create_preset(self.stream_name, self.port_number, self.networking_interface,
                                       data_type=self.data_type, num_channels=num_channels)  # update preset in settings
        self.create_buffer()  # recreate the interface and buffer, using the new preset
        self.worker.reset_interface(self.stream_name, get_stream_preset_info(self.stream_name, 'channel_names'))

        # self.stream_options_window.reload_preset_to_UI()
        # self.reset_viz()

    # def reset_viz(self):
    #     """
    #     caller to this function must ensure self.group_info is modified and up to date with user changes
    #     """
    #     self.clear_stream_visualizations()
    #     self.create_visualization_component()

    def create_buffer(self):
        channel_names = get_stream_preset_info(self.stream_name, 'channel_names')
        buffer_size = 1 if len(
            channel_names) > config.MAX_TIMESERIES_NUM_CHANNELS_PER_STREAM else config.VIZ_DATA_BUFFER_MAX_SIZE
        self.viz_data_buffer = DataBufferSingleStream(num_channels=len(channel_names),
                                                      buffer_sizes=buffer_size, append_zeros=True)

    def remove_stream(self):

        if self.main_parent.recording_tab.is_recording:
            self.main_parent.current_dialog = dialog_popup(msg='Cannot remove stream while recording.')
            return False
        self.timer.stop()
        self.v_timer.stop()
        if self.worker.is_streaming:
            self.worker.stop_stream()
        self.worker_thread.exit()
        self.worker_thread.wait()  # wait for the thread to exit

        self.main_parent.stream_widgets.pop(self.stream_name)
        self.main_parent.remove_stream_widget(self)
        # close window if popped
        if self.is_popped:
            self.delete_window()
        self.deleteLater()
        self.stream_options_window.close()
        return True

    def process_stream_data(self, data_dict):
        pass
    #     '''
    #     update the visualization buffer, recording buffer, and scripting buffer
    #     '''
    #     if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:  # if there are data in the emitted data dict
    #         try:
    #             self.run_data_processor(data_dict)
    #             self.viz_data_head = self.viz_data_head + len(data_dict['timestamps'])
    #             self.update_buffer_times.append(timeit(self.viz_data_buffer.update_buffer, (data_dict,))[
    #                                                 1])  # NOTE performance test scripts, don't include in production code
    #             self._has_new_viz_data = True
    #             # self.viz_data_buffer.update_buffer(data_dict)
    #         except ChannelMismatchError as e:
    #             self.in_error_state = True
    #             preset_chan_num = len(get_stream_preset_info(self.stream_name, 'channel_names'))
    #             message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
    #             reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent,
    #                                  buttons=self.channel_mismatch_buttons)
    #
    #             if reply.result():
    #                 self.reset_preset_by_num_channels(e.message)
    #                 self.in_error_state = False
    #                 return
    #             else:
    #                 self.start_stop_stream_btn_clicked()  # stop the stream
    #                 self.in_error_state = False
    #                 return
    #         self.actualSamplingRate = data_dict['sampling_rate']
    #         self.current_timestamp = data_dict['timestamps'][-1]
    #         # notify the internal buffer in recordings tab
    #
    #         # reshape data_dict based on sensor interface
    #         self.main_parent.recording_tab.update_recording_buffer(data_dict)
    #         self.main_parent.scripting_tab.forward_data(data_dict)
    #         # scripting tab
    #
    # '''
    # settings on change:
    # visualization can be changed while recording with mutex
    # 1. lock settings on change
    # 2. update visualization
    # 3. save changes to RENA_Settings
    # 4. release mutex
    #
    # data processing cannot be changed while recording
    #
    # # cannot add channels while streaming/recording
    #
    #
    # '''

    # def stream_settings_changed(self, change):
    #     self.setting_update_viz_mutex.lock()
    #     # resolve the
    #     if change[0] == "nominal_sampling_rate":
    #         pass  # TODO
    #     # TODO add other changes such as plot format, plot order, etc...
    #
    #     self.setting_update_viz_mutex.unlock()

    def visualize(self):
        '''
        This is the function for LSL data visualization.
        It plot the data from the data visualization buffer based on the configuration
        The data to plot is in the parameter self.viz_data_buffer
        '''
        pass

        # self.tick_times.append(time.time())
        # # print("Viz FPS {0}".format(self.get_fps()), end='\r')
        # self.worker.signal_stream_availability_tick.emit()  # signal updating the stream availability
        # # for lsl_stream_name, data_to_plot in self.LSL_data_buffer_dicts.items():
        # actual_sampling_rate = self.actualSamplingRate
        #
        # if not self._has_new_viz_data:
        #     return
        # self.viz_data_buffer.buffer[0][np.isnan(self.viz_data_buffer.buffer[0])] = 0  # zero out nan
        #
        # if AppConfigs().linechart_viz_mode == LinechartVizMode.INPLACE:
        #     data_to_plot = self.viz_data_buffer.buffer[0][:, -self.viz_data_head:]
        # elif AppConfigs().linechart_viz_mode == LinechartVizMode.CONTINUOUS:
        #     data_to_plot = self.viz_data_buffer.buffer[0][:, -self.num_points_to_plot:]
        # for plot_group_index, (group_name) in enumerate(get_stream_group_info(self.stream_name).keys()):
        #     self.plot_data_times.append(timeit(self.viz_components.group_plots[group_name].plot_data, (data_to_plot,))[
        #                                     1])  # NOTE performance test scripts, don't include in production code
        #     # self.viz_components.group_plots[group_name].plot_data(data_to_plot)
        #
        # # show the label
        # self.viz_components.fs_label.setText(
        #     'Sampling rate = {:.3f}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        # self.viz_components.ts_label.setText('Current Time Stamp = {:.3f}'.format(self.current_timestamp))
        #
        # self._has_new_viz_data = False
        # if self.viz_data_head > get_stream_preset_info(self.stream_name, 'display_duration') * get_stream_preset_info(
        #         self.stream_name, 'nominal_sampling_rate'):  # reset the head if it is out of bound
        #     self.viz_data_head = 0

    def ticks(self):
        self.worker.signal_data_tick.emit()



    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0

    def is_widget_streaming(self):
        return self.worker.is_streaming


    def try_close(self):
        return self.remove_stream()



