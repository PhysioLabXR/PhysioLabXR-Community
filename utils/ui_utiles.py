from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QLabel, QCheckBox, QFrame, QVBoxLayout, QHBoxLayout, QComboBox

import config
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

def init_container(parent, label=None, label_position=None, label_bold=True, vertical=True, style=None, size=None):
    container = QtGui.QWidget()

    if size:
        container.setFixedWidth(size[0])
        container.setFixedHeight(size[1])

    if style:  # set the style of the container, which takes over the invisible layout
        container.setStyleSheet(style)

    parent.addWidget(container)

    vl = init_view(label, container, label_bold, label_position, vertical)

    return vl


def init_button(parent, label=None, function=None, style=config_ui.button_style_classic):
    btn = QtWidgets.QPushButton(text=label)
    if function:
        btn.clicked.connect(function)
    parent.addWidget(btn)
    btn.setStyleSheet(config_ui.button_style_classic)

    return btn

def resolve_sensor_label(sensor_type):
    sensor_widget_label = ''
    if sensor_type == config.sensors[0]:
        sensor_widget_label = 'OpenBCI Cyton'
    elif sensor_type == config.sensors[1]:
        sensor_widget_label = 'Vive Pro eye-tracking (Unity)'
    return sensor_widget_label

def init_sensor_widget(parent, sensor_type):
    sensor_widget_label = resolve_sensor_label(sensor_type)

    sensor_layout = init_container(parent=parent, label=sensor_widget_label)
    start_stream_btn = init_button(parent=sensor_layout, label='Start Stream')
    stop_stream_btn = init_button(parent=sensor_layout, label='Stop Stream')
    return sensor_layout, start_stream_btn, stop_stream_btn