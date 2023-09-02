# This Python file uses the following encoding: utf-8

from PyQt6.QtCore import QThread, QMutex

from physiolabxr.exceptions.exceptions import LSLStreamNotFoundError, ChannelMismatchError
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.examples.fmri_experiment_example.mri_utils import *
# get_mri_coronal_view_dimension, get_mri_sagittal_view_dimension, \
#     get_mri_axial_view_dimension
from physiolabxr.presets.presets_utils import get_stream_preset_info, set_stream_num_channels, get_stream_num_channels, \
    get_fmri_data_shape
from physiolabxr.threadings import workers
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel
# This Python file uses the following encoding: utf-8
import time
from collections import deque

import numpy as np
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QPixmap

from physiolabxr.configs import config_ui
from physiolabxr.ui.PoppableWidget import Poppable

from physiolabxr.utils.buffers import DataBufferSingleStream
from physiolabxr.utils.ui_utils import dialog_popup


class FMRIWidget(Poppable, QtWidgets.QWidget):
    def __init__(self, parent_widget, parent_layout, stream_name, data_type, worker,
                 insert_position=None):
        """

        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(stream_name, parent_widget, parent_layout, self.remove_stream)

        self.ui = uic.loadUi("examples/fmri_experiment_example/FMRIWidgetNew._ui", self)
        self.setWindowTitle('fMRI Viewer')

        self.create_visualization_component()
        self.load_mri_volume()
        self.init_fmri_gl_axial_view_image_item()

        self.parent = parent_layout
        self.main_parent = parent_widget

        self.set_pop_button(self.PopWindowBtn)
        self.stream_name = stream_name
        self.data_type = data_type

        self.actualSamplingRate = 0

        self.StreamNameLabel.setText(stream_name)
        self.StartStopStreamBtn.setIcon(AppConfigs()._icon_start)
        self.OptionsBtn.setIcon(AppConfigs._icon_options)
        self.RemoveStreamBtn.setIcon(AppConfigs()._icon_remove_stream)

        self.is_stream_available = False
        self.in_error_state = False  # an error state to prevent ticking when is set to true
        # visualization data buffer
        self.current_timestamp = 0
        self.fmri_viz_volume = np.zeros((240, 240, 186))
        self._has_new_viz_data = False

        self.viz_data_buffer = None
        self.create_buffer()

        # timer
        self.timer = QTimer()
        self.timer.setInterval(AppConfigs().pull_data_interval)
        self.timer.timeout.connect(self.ticks)

        # visualization timer
        self.v_timer = QTimer()
        self.v_timer.setInterval(AppConfigs().visualization_refresh_interval)
        self.v_timer.timeout.connect(self.visualize)

        # connect btn
        self.StartStopStreamBtn.clicked.connect(self.start_stop_stream_btn_clicked)
        self.OptionsBtn.clicked.connect(self.options_btn_clicked)
        self.RemoveStreamBtn.clicked.connect(self.remove_stream)

        # inefficient loading of assets TODO need to confirm creating Pixmap in ui_shared result in crash
        self.stream_unavailable_pixmap = QPixmap(AppConfigs()._stream_unavailable)
        self.stream_available_pixmap = QPixmap(AppConfigs()._stream_available)
        self.stream_active_pixmap = QPixmap(AppConfigs()._stream_viz_active)

        # init worker thread
        self.worker_thread = QThread(self)
        self.worker = workers.ZMQWorker(port_number=5559, subtopic='fMRI', data_type=self.data_type.value)
        self.worker.signal_data.connect(self.process_stream_data)
        self.worker.signal_stream_availability.connect(self.update_stream_availability)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.tick_times = deque(maxlen=10 * AppConfigs().visualization_refresh_interval)
        self.setting_update_viz_mutex = QMutex()

        # start the timers

        self.timer.start()
        self.v_timer.start()
        # self.scaling_factor = (0.9375, 0.9375, 1.5)
        #
        #
        # self.sagittal_view_transformation = QTransform()
        # self.sagittal_view_transformation.scale(0.9375, 1.5)
        #
        # self.coronal_view_transformation = QTransform()
        # self.coronal_view_transformation.scale(0.9375, 1.5)
        #
        # self.axial_view_transformation = QTransform()
        # self.axial_view_transformation.scale(0.9375, 0.9375)
        #


        self.__post_init__()

    def __post_init__(self):
        self.sagittal_view_slider_value = 0
        self.coronal_view_slider_value = 0
        self.axial_view_slider_value = 0

    def create_visualization_component(self):
        ######################################################
        self.volume_view_plot = gl.GLViewWidget()
        self.VolumnViewPlotWidget.layout().addWidget(self.volume_view_plot)

        ######################################################
        self.sagittal_view_plot = pg.PlotWidget()
        self.sagittal_view_plot.setTitle("Sagittal View")
        self.sagittal_view_plot.setLabel(axis='left', text='millimeters')
        self.sagittal_view_plot.setLabel(axis='bottom', text='millimeters')
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
        self.coronal_view_plot.setTitle("Coronal View")
        self.coronal_view_plot.setLabel(axis='left', text='millimeters')
        self.coronal_view_plot.setLabel(axis='bottom', text='millimeters')
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
        self.axial_view_plot.setTitle("Axial View")
        self.axial_view_plot.setLabel(axis='left', text='millimeters')
        self.axial_view_plot.setLabel(axis='bottom', text='millimeters')
        self.AxiaViewPlotWidget.layout().addWidget(self.axial_view_plot)
        self.axial_view_mri_image_item = pg.ImageItem()
        self.axial_view_fmri_image_item = pg.ImageItem()
        self.axial_view_plot.addItem(self.axial_view_mri_image_item)
        self.axial_view_plot.addItem(self.axial_view_fmri_image_item)

        self.axial_view_slider = SliderWithValueLabel()
        self.axial_view_slider.valueChanged.connect(self.axial_view_slider_on_change)
        self.AxiaViewSliderWidget.layout().addWidget(self.axial_view_slider)

    # def init_fmri_graphic_component(self):
    #     self.fmri_timestamp_slider = SliderWithValueLabel()
    #     self.fmri_timestamp_slider.valueChanged.connect(self.fmri_timestamp_slider_on_change)
    #     self.FMRITimestampSliderWidget.layout().addWidget(self.fmri_timestamp_slider)

    def load_mri_volume(self):
        # g = gl.GLGridItem()
        # g.scale(100, 100, 100)
        # self.volume_view_plot.addItem(g)

        _, self.mri_volume_data = load_nii_gz_file(
            'D:/HaowenWei/Rena/RenaLabApp/physiolabxr/examples/fmri_experiment_example/structural.nii.gz', zoomed=True)
        self.gl_volume_item = volume_to_gl_volume_item(self.mri_volume_data, non_linear_interpolation_factor=2)
        self.volume_view_plot.addItem(self.gl_volume_item)
        self.set_mri_view_slider_range()

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
        fmri_slice = get_mri_coronal_view_slice(self.fmri_viz_volume, self.coronal_view_slider_value)
        self.coronal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.55))

    def set_sagittal_view_fmri(self):
        fmri_slice = get_mri_sagittal_view_slice(self.fmri_viz_volume, self.sagittal_view_slider_value)
        self.sagittal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.55))

    def set_axial_view_fmri(self):
        fmri_slice = get_mri_axial_view_slice(self.fmri_viz_volume, self.axial_view_slider_value)
        self.axial_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.55))
        image_data = (gray_to_heatmap(fmri_slice, threshold=0.55)*255).astype(np.uint8)
        # image_data = np.transpose(image_data, (1, 0, 2))
        self.fmri_axial_view_image_item.setData(image_data)



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
        # self.main_parent.update_active_streams()

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

    def process_stream_data(self, data_dict):
        # set visualization buffer and set has data flag

        if data_dict['frames'].shape[-1] > 0 and not self.in_error_state:

            self.viz_data_buffer.update_buffer(data_dict)
            self.actualSamplingRate = data_dict['sampling_rate']
            self.current_timestamp = data_dict['timestamps'][-1]
            self._has_new_viz_data = True
        else:
            self._has_new_viz_data = False

    def create_buffer(self):
        channel_num = get_stream_num_channels(self.stream_name)
        # buffer_size = 1 if channel_num > config.MAX_TIMESERIES_NUM_CHANNELS_PER_STREAM else config.VIZ_DATA_BUFFER_MAX_SIZE
        # self.viz_data_buffer = DataBufferSingleStream(num_channels=len(channel_names), buffer_sizes=buffer_size, append_zeros=True)

        self.viz_data_buffer = DataBufferSingleStream(num_channels=channel_num,
                                                      buffer_sizes=1, append_zeros=True)

    def visualize(self):
        self.tick_times.append(time.time())
        self.worker.signal_stream_availability_tick.emit()
        actual_sampling_rate = self.actualSamplingRate

        if not self._has_new_viz_data:
            return

        self.update_fmri_visualization()

        self.fs_label.setText(
            'Sampling rate = {:.3f}'.format(round(actual_sampling_rate, config_ui.sampling_rate_decimal_places)))
        self.ts_label.setText('Current Time Stamp = {:.3f}'.format(self.current_timestamp))
        self._has_new_viz_data = False

    def update_fmri_visualization(self):
        if self.viz_data_buffer.has_data():
            fmri_data = self.viz_data_buffer.buffer[0][:, -1]
            data_shape = get_fmri_data_shape(self.stream_name)
            self.fmri_viz_volume = np.reshape(fmri_data, data_shape)

            # self.fmri_viz_volume.normalize()
            self.set_axial_view_fmri()
            self.set_sagittal_view_fmri()
            self.set_coronal_view_fmri()

    def ticks(self):
        self.worker.signal_data_tick.emit()

    def is_streaming(self):
        return self.worker.is_streaming

    def start_stop_stream_btn_clicked(self):
        if self.worker.is_streaming:
            self.worker.stop_stream()
            if not self.worker.is_streaming:
                self.update_stream_availability(self.worker.is_stream_available)
        else:
            try:
                self.worker.start_stream()
            except LSLStreamNotFoundError as e:
                self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
                return

            except ChannelMismatchError as e:
                preset_chan_num = len(get_stream_preset_info(self.stream_name, 'num_channels'))
                message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
                reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent,
                                     buttons=self.channel_mismatch_buttons)

                if reply.result():
                    self.reset_preset_by_num_channels(e.message)
                    self.worker.start_stream()

        self.set_button_icons()

    def reset_preset_by_num_channels(self, num_channels):
        set_stream_num_channels(self.stream_name, num_channels)

    def get_fps(self):
        try:
            return len(self.tick_times) / (self.tick_times[-1] - self.tick_times[0])
        except (ZeroDivisionError, IndexError) as e:
            return 0

    def is_widget_streaming(self):
        return self.worker.is_streaming

    def remove_stream(self):
        pass

    def try_close(self):
        return self.remove_stream()

    def init_fmri_gl_axial_view_image_item(self):
        # image_data = np.random.randint(0, 256, (256, 256, 4), dtype=np.uint8)
        image_data = np.zeros((240, 240, 4), dtype=np.uint8)
        self.fmri_axial_view_image_item = gl.GLImageItem(image_data) #np.zeros((256, 256, 4), dtype=np.uint8)
        self.fmri_axial_view_image_item.scale(1, 1, 1)

        # apply the xz plane transform
        self.fmri_axial_view_image_item.translate(-240 / 2, -240 / 2 , -186/2+105+0.1) #

        self.volume_view_plot.addItem(self.fmri_axial_view_image_item)

