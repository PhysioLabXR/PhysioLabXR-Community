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

def init_inputBox(parent, label=None, label_bold=False, default_input=None):
    container, layout = init_container(parent=parent,
                           label=label,
                           label_bold=label_bold,
                           vertical=False)
    textbox = QtWidgets.QLineEdit()
    textbox.setContentsMargins(5, 0, 0, 0)
    textbox.setText(str(default_input))
    layout.addWidget(textbox)
    textbox.setStyleSheet("background-color:white;")

    return layout, textbox

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


def init_sensor_or_lsl_widget(parent, label_string, insert_position):
    container_widget, layout = init_container(parent=parent, label=config_ui.sensors_type_ui_name_dict[label_string] if label_string in config_ui.sensors_type_ui_name_dict.keys() else label_string, insert_position=insert_position)
    start_stream_btn = init_button(parent=layout, label='Start Stream')
    stop_stream_btn = init_button(parent=layout, label='Stop Stream')
    return container_widget, layout, start_stream_btn, stop_stream_btn


def init_add_widget(parent, lsl_presets: dict):
    container, layout = init_container(parent=parent, label='Add Sensor or LSL', label_bold=True)

    container_add_sensor, layout_add_sensor = init_container(parent=layout, label='Select a Stream to Add', vertical=False)
    sensor_combo_box = init_combo_box(parent=layout_add_sensor, label=None, item_list=list(config_ui.sensors_type_ui_name_dict.values()) + list(lsl_presets.keys()))


    add_sensor_btn = init_button(parent=layout_add_sensor, label='Add')

    container_add_lsl, layout_add_lsl = init_container(parent=layout, label='Define a Stream to Add', vertical=False)
    _, lsl_data_type_input = init_inputBox(parent=layout_add_lsl, default_input=config_ui.default_add_lsl_data_type)
    _, lsl_num_chan_input = init_inputBox(parent=layout_add_lsl, default_input=1)
    add_lsl_btn = init_button(parent=layout_add_lsl, label='Add')

    return layout, sensor_combo_box, add_sensor_btn, lsl_data_type_input, lsl_num_chan_input, add_lsl_btn


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

def dialog_popup(msg):
    dlg = CustomDialog(msg)  # If you pass self, the dialog will be centered over the main window as before.
    if dlg.exec_():
        print("Success!")
    else:
        print("Cancel!")