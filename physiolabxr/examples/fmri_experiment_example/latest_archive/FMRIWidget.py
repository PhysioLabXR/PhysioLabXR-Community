# This Python file uses the following encoding: utf-8

from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QTimer, QThread

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.examples.fmri_experiment_example.mri_utils import *
# get_mri_coronal_view_dimension, get_mri_sagittal_view_dimension, \
#     get_mri_axial_view_dimension
from physiolabxr.presets.PresetEnums import DataType
from physiolabxr.threadings.workers import ZMQWorker
from physiolabxr.ui.PoppableWidget import Poppable
from physiolabxr.ui.SliderWithValueLabel import SliderWithValueLabel


class FMRIWidget(Poppable, QtWidgets.QWidget):
    def __init__(self, parent_widget, parent_layout, window_title,
                 insert_position=None):
        """

        @param parent_widget:
        @param parent_layout:
        @param video_device_name:
        @param insert_position:
        """
        super().__init__(window_title, parent_widget, parent_layout, self.remove_function)

        self.ui = uic.loadUi("examples/fmri_experiment_example/FMRIWidget._ui", self)
        self.set_pop_button(self.PopWindowBtn)

        self.OptionsBtn.setIcon(AppConfigs()._icon_options)
        self.RemoveVideoBtn.setIcon(AppConfigs()._icon_remove_stream)

        self.fmri_timestamp_slider_value = 0

        self.init_mri_graphic_components()
        self.init_fmri_graphic_component()

        self.load_mri_volume()
        self.load_fmri_volume()

        self.timer = QTimer()
        self.worker_thread = QThread(self)
        self.worker = ZMQWorker(port_number=5559, subtopic='fMRI', data_type=DataType.float64)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.__post_init__()

    def __post_init__(self):
        self.sagittal_view_slider_value = 0
        self.coronal_view_slider_value = 0
        self.axial_view_slider_value = 0
        self.fmri_timestamp_slider_value = 0

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
        self.fmri_timestamp_slider.valueChanged.connect(self.fmri_timestamp_slider_on_change)
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

    def load_fmri_volume(self):
        _, self.fmri_volume_data = load_nii_gz_file(
            'C:/Users/Haowe/OneDrive/Desktop/Columbia/RENA/RealityNavigation/physiolabxr/examples/fmri_experiment_example/resampled_fmri.nii.gz')
        self.set_fmri_view_slider_range()
        # self.volume_view_plot.setCameraPosition(distance=200)
        # g = gl.GLGridItem()
        # g.scale(100, 100, 100)
        # self.volume_view_plot.addItem(g)

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
        fmri_slice = get_fmri_coronal_view_slice(self.fmri_volume_data, self.coronal_view_slider_value,
                                                 self.fmri_timestamp_slider_value)
        self.coronal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))

    def set_sagittal_view_fmri(self):
        fmri_slice = get_fmri_coronal_view_slice(self.fmri_volume_data, self.sagittal_view_slider_value,
                                                 self.fmri_timestamp_slider_value)
        self.sagittal_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))

    def set_axial_view_fmri(self):
        fmri_slice = get_fmri_axial_view_slice(self.fmri_volume_data, self.axial_view_slider_value,
                                               self.fmri_timestamp_slider_value)
        self.axial_view_fmri_image_item.setImage(gray_to_heatmap(fmri_slice, threshold=0.5))

    def fmri_timestamp_slider_on_change(self):
        self.fmri_timestamp_slider_value = self.fmri_timestamp_slider.value()

        self.set_coronal_view_fmri()
        self.set_sagittal_view_fmri()
        self.set_axial_view_fmri()

    def process_stream_data(self, data_dict):
        pass
        # if data_dict['frames'].shape[-1] > 0:
        #     self.fmri_volume_data = data_dict

    def remove_function(self):
        pass

