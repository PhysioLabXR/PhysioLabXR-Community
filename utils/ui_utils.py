from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QLabel, QCheckBox, QFrame, QVBoxLayout, QHBoxLayout, QComboBox, QDialog, QDialogButtonBox

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

def init_container(parent, label=None, label_position=None, label_bold=True, vertical=True, style=None, size=None, insert_position=None):
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


def init_button(parent, label=None, function=None, style=config_ui.button_style_classic):
    btn = QtWidgets.QPushButton(text=label)
    if function:
        btn.clicked.connect(function)
    parent.addWidget(btn)
    btn.setStyleSheet(config_ui.button_style_classic)

    return btn

def init_combo_box(parent, label, item_list):
    container_widget, vl = init_container(parent=parent, label=label, vertical=False)
    combo_widget = QtGui.QWidget()
    combo_box = QComboBox()
    for i in item_list:
        combo_box.addItem(i)
    vl.addWidget(combo_box)

    return combo_box


def init_sensor_widget(parent, sensor_type, insert_position):
    container_widget, sensor_layout = init_container(parent=parent, label=config_ui.sensors_type_ui_name_dict[sensor_type], insert_position=insert_position)
    start_stream_btn = init_button(parent=sensor_layout, label='Start Stream')
    stop_stream_btn = init_button(parent=sensor_layout, label='Stop Stream')
    return container_widget, sensor_layout, start_stream_btn, stop_stream_btn


def init_add_sensor_widget(parent):
    container_widget, add_sensor_layout = init_container(parent=parent, label='Select a Sensor to Add', label_bold=True)
    sensor_combo_box = init_combo_box(parent=add_sensor_layout, label=None, item_list=list(config_ui.sensors_type_ui_name_dict.values()))
    add_btn = init_button(parent=add_sensor_layout, label='Add Sensor')
    return add_sensor_layout, sensor_combo_box, add_btn


class CustomDialog(QDialog):
    def __init__(self, msg, parent=None):
        super().__init__(parent=parent)

        self.setWindowTitle("Warning")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(msg)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)