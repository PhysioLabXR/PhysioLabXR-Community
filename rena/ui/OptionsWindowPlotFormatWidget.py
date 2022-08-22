# This Python file uses the following encoding: utf-8

from PyQt5 import uic
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import QDialog, QTreeWidget, QLabel, QTreeWidgetItem

from rena import config_signal, config
from rena.config_ui import *
from rena.ui.StreamGroupView import StreamGroupView
from rena.utils.settings_utils import collect_stream_all_groups_info, collect_stream_group_info, \
    collect_stream_group_plot_format, set_image_plot_format
from rena.utils.ui_utils import init_container, init_inputBox, dialog_popup, init_label, init_button, init_scroll_label
from PyQt5 import QtCore, QtGui, QtWidgets


class OptionsWindowPlotFormatWidget(QtWidgets.QWidget):
    plot_format_on_change = QtCore.pyqtSignal(str)

    def __init__(self, stream_name, selected_group_name):
        super().__init__()
        """
        :param lsl_data_buffer: dict, passed by reference. Do not modify, as modifying it makes a copy.
        :rtype: object
        """
        # self.setWindowTitle('Options')
        self.ui = uic.loadUi("ui/OptionsWindowPlotFormatWidget.ui", self)

        self.stream_name = stream_name
        self.selected_group_name = selected_group_name

        self.TimeSeriesCheckBox.stateChanged.connect(lambda:self.TimeSeriesCheckBox_status_change(self.TimeSeriesCheckBox))
        self.ImageCheckBox.stateChanged.connect(lambda:self.ImageCheckBox_status_change(self.ImageCheckBox))
        self.BarPlotCheckbox.stateChanged.connect(lambda:self.BarPlotCheckbox_status_change(self.BarPlotCheckbox))

        self.group_info = collect_stream_group_info(stream_name=self.stream_name, group_name=self.selected_group_name)
        self.init_plot_format_info()




        # self.updatePlotFormatBtn.clicked.connect(self.update_plot_format_btn_clicked)

    # plot_format = {
    #     'time_series': {'display': True},
    #     'image': {'display': False,
    #               'format': None,
    #               'width': None,
    #               'height': None,
    #               'depth': None,
    #               },
    #     'bar_plot': {'display': False}
    # }

    def clearLayout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearLayout(item.layout())
    #
    # def set_plot_format(self):
    #     # get plot format
    #     # checkbox
    #     # actions after plot format changed
    #     pass

    def init_plot_format_info(self):
        if self.group_info['plot_format']['time_series']['display']:
            self.TimeSeriesCheckBox.setCheckState(Qt.Checked)
        else:
            self.ImageCheckBox.setCheckState(Qt.Unchecked)
            self.TimeSeiresFormatInfoWidget.setEnabled(False)


        ###################################################################
        if self.group_info['plot_format']['image']['display']:
            self.ImageCheckBox.setCheckState(Qt.Checked)
            # TODO: set size
            image_format = self.group_info['plot_format']['image']
            # if image_format['width'] * image_format['height'] * image_format['depth'] != len(
            #         self.group_info['channel_indices']):
            #     dialog_popup(
            #         'Warning, the preset might be corrupted. The WxHxD not equal to the total number of channel')

            self.ImageWidthTextEdit.setText(str(image_format['width']))
            self.ImageHeightTextEdit.setText(str(image_format['height']))
            self.imageFormatComboBox.setCurrentText(image_format['image_format'])

        else:
            self.ImageCheckBox.setCheckState(Qt.Unchecked)
            self.ImageFormatInfoWidget.setEnabled(False)
        #################################################################

        if self.group_info['plot_format']['bar_plot']['display']:
            self.BarPlotCheckbox.setCheckState(Qt.Checked)
        else:
            self.BarPlotCheckbox.setCheckState(Qt.Unchecked)
            self.BarPlotFormatInfoWidget.setEnabled(False)



    # def update_plot_format_btn_clicked(self):
    #     # get display status
    #     if self.ImageCheckBox.isChecked():
    #         image_width = self.ImageWidthTextEdit.text()
    #         image_height = self.ImageHeightTextEdit.text()
    #         if image_width.isnumeric() == False or image_height.isnumeric() == False:
    #             dialog_popup('Please Enter a positive Integer for Width and Height')
    #             # self.init_plot_format_info()
    #             return
    #
    #         # if image pixel number does  not match
    #         # image_format = self.group_info['plot_format']['image']
    #         image_width=int(image_width)
    #         image_height=int(image_height)
    #         image_depth = image_depth_dict[self.imageFormatComboBox.currentText()]
    #         if image_width * image_height * image_depth != len(self.group_info['channel_indices']):
    #             dialog_popup('Image WxHxD must equal to the total number of channel')
    #             # self.init_plot_format_info()
    #             return





    def TimeSeriesCheckBox_status_change(self, checkbox):
        if checkbox.isChecked():
            pass
            # self.TimeSeiresFormatInfoWidget.setEnabled(True)
        else:
            # self.TimeSeiresFormatInfoWidget.setEnabled(False)
            pass

    def ImageCheckBox_status_change(self, checkbox):
        if checkbox.isChecked():
            #################### check input is valid ################################
                # if self.ImageCheckBox.isChecked():
            image_width = self.ImageWidthTextEdit.text()
            image_height = self.ImageHeightTextEdit.text()
            if image_width.isnumeric() == False or image_height.isnumeric() == False:
                dialog_popup('Please Enter a positive Integer for Width and Height')
                self.ImageCheckBox.setCheckState(Qt.Unchecked)
                return
            # self.ImageFormatInfoWidget.setEnabled(True)
            group_info = collect_stream_group_info(self.stream_name, self.selected_group_name)
            image = group_info['plot_format']['image']

            if int(image_width) * int(image_height) * image_depth_dict[self.imageFormatComboBox.currentText()] != len(
                    self.group_info['channel_indices']):
                dialog_popup(
                    'Warning, the preset might be corrupted. The WxHxD not equal to the total number of channel')
                self.ImageCheckBox.setCheckState(Qt.Unchecked)
                return

            else:
                # update the preset
                set_image_plot_format(self.stream_name, self.selected_group_name, image_height, image_width, self.imageFormatComboBox.currentText())

                pass
                #


            # check channel num



            pass
        else:
            # self.ImageFormatInfoWidget.setEnabled(False)


            pass

    def BarPlotCheckbox_status_change(self, checkbox):
        if checkbox.isChecked():
            # self.BarPlotFormatInfoWidget.setEnabled(True)
            pass
        else:
            # self.BarPlotFormatInfoWidget.setEnabled(False)
            pass

    # def update
    #
    # def updateImageFormatBtnClicked(self):



    # def update_plot_format_dict(self):
    #     #
    #     plot_format = dict()
    #
    #     if self.TimeSeriesCheckBox.isChecked():
    #         plot_format['time_series']['display']=True



        # plot_format = dict()
        # if plot_format = True
        # plot_format['time_series']=bool(self.TimeSeriesCheckBox.checkState())
        # plot_format['time_series'] = bool(self.TimeSeriesCheckBox.checkState())













