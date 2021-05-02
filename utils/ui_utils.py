import cv2
from PyQt5 import QtGui
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt, QFile, QTextStream
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QHBoxLayout, QComboBox, QDialog, QDialogButtonBox, \
    QGraphicsView, QGraphicsScene
from PyQt5.QtWidgets import QLabel, QVBoxLayout
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget

import config_ui


def init_view(label, container, label_bold=True, position="centertop", vertical=True):
    if vertical:
        vl = QVBoxLayout(container)
    else:
        vl = QHBoxLayout(container)
    if label:
        ql = QLabel()
        if label_bold:
            ql.setStyleSheet("font: bold 14px;")

        # positions
        if position == "centertop":
            ql.setAlignment(QtCore.Qt.AlignTop)
            ql.setAlignment(QtCore.Qt.AlignCenter)

        elif position == "center":
            ql.setAlignment(QtCore.Qt.AlignCenter)

        elif position == "rightbottom":
            ql.setAlignment(QtCore.Qt.AlignRight)
            ql.setAlignment(QtCore.Qt.AlignBottom)

        elif position == "righttop":
            ql.setAlignment(QtCore.Qt.AlignRight)
            ql.setAlignment(QtCore.Qt.AlignTop)

        elif position == "lefttop":
            ql.setAlignment(QtCore.Qt.AlignLeft)
            ql.setAlignment(QtCore.Qt.AlignTop)

        elif position == "leftbottom":
            ql.setAlignment(QtCore.Qt.AlignLeft)
            ql.setAlignment(QtCore.Qt.AlignBottom)

        ql.setText(label)
        vl.addWidget(ql)

    return vl


def init_container(parent, label=None, label_position=None, label_bold=True, vertical=True, style=None, size=None,
                   insert_position=None):
    container = QtGui.QWidget()

    if size:
        container.setFixedWidth(size[0])
        container.setFixedHeight(size[1])

    if style:  # set the style of the container, which takes over the invisible layout
        container.setStyleSheet(style)
    if type(insert_position) == int:
        parent.insertWidget(insert_position, container)
    else:
        parent.addWidget(container)

    vl = init_view(label, container, label_bold, label_position, vertical)

    return container, vl


def init_inputBox(parent, label=None, label_bold=False, default_input=None):
    container, layout = init_container(parent=parent,
                                       label=label,
                                       label_bold=label_bold,
                                       vertical=False)
    textbox = QtWidgets.QLineEdit()
    textbox.setContentsMargins(5, 0, 0, 0)
    textbox.setText(str(default_input))
    layout.addWidget(textbox)
    # textbox.setStyleSheet("background-color:white;")

    return layout, textbox


def init_button(parent, label=None, function=None, style=config_ui.button_style_classic):
    btn = QtWidgets.QPushButton(text=label)
    if function:
        btn.clicked.connect(function)
    parent.addWidget(btn)
    # btn.setStyleSheet(config_ui.button_style_classic)

    return btn


def init_combo_box(parent, label, item_list):
    container_widget, vl = init_container(parent=parent, label=label, vertical=False)
    combo_box = QComboBox()
    for i in item_list:
        combo_box.addItem(i)
    vl.addWidget(combo_box)

    return combo_box


def init_camera_widget(parent, label_string, insert_position):
    container_widget, layout = init_container(parent=parent, insert_position=insert_position)
    container_widget.setFixedWidth(120 + max(config_ui.cam_display_width, config_ui.capture_display_width))

    camera_img_label = QLabel()
    _, label_btn_layout = init_container(parent=layout, vertical=False)
    cam_id_label = QLabel(label_string)
    cam_id_label.setStyleSheet("font: bold 14px;")
    label_btn_layout.addWidget(cam_id_label)
    remove_cam_btn = init_button(parent=label_btn_layout, label='Remove Capture')
    layout.addWidget(camera_img_label)

    return container_widget, layout, remove_cam_btn, camera_img_label


def init_spec_view(parent, label, graph=None):
    if label:
        ql = QLabel()
        ql.setAlignment(QtCore.Qt.AlignTop)
        ql.setAlignment(QtCore.Qt.AlignCenter)
        ql.setText(label)
        parent.addWidget(ql)

    spc_gv = QGraphicsView()
    parent.addWidget(spc_gv)

    scene = QGraphicsScene()
    spc_gv.setScene(scene)
    spc_gv.setAlignment(QtCore.Qt.AlignCenter)
    if graph:
        scene.addItem(graph)
    # spc_gv.setFixedSize(config.WINDOW_WIDTH/4, config.WINDOW_HEIGHT/4)
    return scene


def init_sensor_or_lsl_widget(parent, label_string, insert_position):
    container_widget, layout = init_container(parent=parent, insert_position=insert_position)

    _, top_layout = init_container(parent=layout, vertical=False)
    ql = QLabel(config_ui.sensors_type_ui_name_dict[
                    label_string] if label_string in config_ui.sensors_type_ui_name_dict.keys() else label_string)
    ql.setStyleSheet("font: bold 14px;")
    signal_settings_btn = init_button(parent=top_layout, label='Signal Settings')
    pop_window_btn = init_button(parent=top_layout, label='Pop Window')

    signal_settings_btn.setFixedWidth(200)
    pop_window_btn.setFixedWidth(200)


    top_layout.addWidget(ql)
    top_layout.addWidget(signal_settings_btn)
    top_layout.addWidget(pop_window_btn)


    start_stream_btn = init_button(parent=layout, label='Start Stream')
    stop_stream_btn = init_button(parent=layout, label='Stop Stream')
    return container_widget, layout, start_stream_btn, stop_stream_btn, pop_window_btn, signal_settings_btn


def init_add_widget(parent, lsl_presets: dict, device_presets: dict):
    container, layout = init_container(parent=parent, label='Add Stream', label_bold=True)
    container.setFixedWidth(700)

    container_add_camera, layout_add_camera = init_container(parent=layout,
                                                             label='Select a Camera(ID) or Screen Capture to add',
                                                             vertical=False)
    # detect camera
    cameras = get_working_camera_id()
    cameras = list(map(str, cameras))
    camera_screen_cap_list = cameras + ['monitor1']
    # add camera container
    camera_combo_box = init_combo_box(parent=layout_add_camera, label=None,
                                      item_list=camera_screen_cap_list)
    add_camera_btn = init_button(parent=layout_add_camera, label='Add')

    reload_presets_btn = init_button(parent=layout, label='Reload Presets')
    # add sensor container
    container_add_sensor, layout_add_sensor = init_container(parent=layout, label='Select a Stream to Add',
                                                             vertical=False)
    sensor_combo_box = init_combo_box(parent=layout_add_sensor, label=None,
                                      item_list=list(
                                          lsl_presets.keys()))

    add_sensor_btn = init_button(parent=layout_add_sensor, label='Add Stream')

    container_connect_device, layout_connect_device = init_container(parent=layout, label='Select a Device to Connect', vertical=False)
    container_connect_device, layout_connect_device = init_container(parent=layout, label='Select a Device to Connect', vertical=False)
    device_combo_box = init_combo_box(parent=layout_connect_device, label=None,
                                      item_list=list(
                                          device_presets.keys()))
    add_preset_device_btn = init_button(parent=layout_connect_device, label='Add Device')

    container_add_lsl, layout_add_lsl = init_container(parent=layout, label='Define a Stream to Add', vertical=False)
    _, lsl_data_type_input = init_inputBox(parent=layout_add_lsl, default_input=config_ui.default_add_lsl_data_type)
    add_lsl_btn = init_button(parent=layout_add_lsl, label='Add')

    return layout, camera_combo_box, add_camera_btn, sensor_combo_box, add_sensor_btn, lsl_data_type_input, add_lsl_btn, reload_presets_btn, device_combo_box, add_preset_device_btn


class CustomDialog(QDialog):
    def __init__(self, title, msg, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(str(msg))
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


def dialog_popup(msg, title='Warning'):
    dlg = CustomDialog(title, msg)  # If you pass self, the dialog will be centered over the main window as before.
    if dlg.exec_():
        print("Dialog popup")
    else:
        print("Cancel!")


import numpy as np
import colorsys


def get_distinct_colors(num_colors, depth=8):
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    colors = [tuple(v * (2 ** depth - 1) for v in c) for c in colors]
    return colors


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    # p = convert_to_Qt_format.scaled(config_ui.cam_display_width, config_ui.cam_display_height, Qt.KeepAspectRatio)
    return QPixmap.fromImage(convert_to_Qt_format)


def get_working_camera_id():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr

def stream_stylesheet(stylesheet_url):
    stylesheet = QFile(stylesheet_url)
    stylesheet.open(QFile.ReadOnly | QFile.Text)
    stream = QTextStream(stylesheet)
    QtWidgets.qApp.setStyleSheet(stream.readAll())


class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, widget_to_add: QWidget, close_function):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(widget_to_add)
        self.close_function = close_function
        self.setLayout(layout)

    def closeEvent(self, event):
        # do stuff
        print('Window closed')
        if self.close_function():
            event.accept()  # let the window close
        else:
            event.ignore()