# This Python file uses the following encoding: utf-8

from PyQt5 import QtWidgets, uic

from rena.ui.PoppableWidget import Poppable
from rena.ui.SliderWithValueLabel import SliderWithValueLabel
from rena.ui_shared import remove_stream_icon, \
    options_icon
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QOpenGLWidget
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from OpenGL.GL import *
from OpenGL.GLUT import *

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph import functions as fn
import nibabel as nib


def load_nii_gz_file(file_path: str, normalized=True):
    nifti_image = nib.load(file_path)

    # Access the image data and header
    image_data = nifti_image.get_fdata()
    image_header = nifti_image.header

    # Convert the image data to a NumPy array
    volume_data = np.array(image_data)

    if normalized:
        (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data))

    return image_header, volume_data


def volume_to_gl_volume_item(volume_data, alpha_interpolate=True, centralized=True):
    if alpha_interpolate:
        alpha_channel = np.interp(volume_data, (0, 1), (0, 255))
    else:
        alpha_channel = 225
    # Add an alpha channel to the volume data
    volume_data_rgba = np.zeros(volume_data.shape + (4,), dtype=np.ubyte)
    volume_data_rgba[..., 0] = volume_data * 255  # R channel
    volume_data_rgba[..., 1] = volume_data * 255  # G channel
    volume_data_rgba[..., 2] = volume_data * 255  # B channel

    volume_data_rgba[..., 3] = alpha_channel  # Alpha channel
    gl_volume_item = gl.GLVolumeItem(data=volume_data_rgba)

    if centralized:
        x_size, y_size, z_size = volume_data.shape
        gl_volume_item.translate(-x_size / 2, -y_size / 2, -z_size / 2)

    return gl_volume_item


# def volume_to_gl_volume_item(volume_data:np.ndarray, alpha_interpretation=False):
#
#
#     volume_data_rgba = np.zeros(volume_data.shape + (4,), dtype=np.ubyte)
#     volume_data_rgba[..., 0] = volume_data * 255  # R channel
#     volume_data_rgba[..., 1] = volume_data * 255  # G channel
#     volume_data_rgba[..., 2] = volume_data * 255  # B channel
#
#     volume_data_rgba[..., 3] = 225  # Alpha channel
#
#     gl_volume_item = gl.GLVolumeItem(data=volume_data_rgba)
#     return gl_volume_item


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

        self.ui = uic.loadUi("examples/fmri_experiment_example/FMRIWidget.ui", self)
        self.set_pop_button(self.PopWindowBtn)

        self.OptionsBtn.setIcon(options_icon)
        self.RemoveVideoBtn.setIcon(remove_stream_icon)

        self.init_graphic_components()
        self.volume_data = None
        self.display_volume()

    def init_graphic_components(self):
        self.volume_view_plot = gl.GLViewWidget()
        self.VolumnViewPlotWidget.layout().addWidget(self.volume_view_plot)

        self.sagittal_view_plot = pg.PlotWidget()
        self.SagittalViewPlotWidget.layout().addWidget(self.sagittal_view_plot)
        self.sagital_view_slider = SliderWithValueLabel()
        self.SagittalViewSliderWidget.layout().addWidget(self.sagital_view_slider)

        self.coronal_view_plot = pg.PlotWidget()
        self.CoronalViewPlotWidget.layout().addWidget(self.coronal_view_plot)
        self.coronal_view_slider = SliderWithValueLabel()
        self.CoronalViewSliderWidget.layout().addWidget(self.coronal_view_slider)

        self.axial_view_plot = pg.PlotWidget()
        self.AxiaViewPlotWidget.layout().addWidget(self.axial_view_plot)
        self.axial_view_slider = SliderWithValueLabel()
        self.AxiaViewSliderWidget.layout().addWidget(self.axial_view_slider)

    def display_volume(self):
        _, self.volume_data = load_nii_gz_file('C:/Users/Haowe/OneDrive/Desktop/Columbia/RENA/RealityNavigation/rena/examples/fmri_experiment_example/structural_brain.nii.gz')
        self.gl_volume_item = volume_to_gl_volume_item(self.volume_data)
        self.volume_view_plot.addItem(self.gl_volume_item)
        self.volume_view_plot.setCameraPosition(distance=200)
        g = gl.GLGridItem()
        g.scale(100, 100, 100)
        self.volume_view_plot.addItem(g)



    def remove_function(self):
        pass
