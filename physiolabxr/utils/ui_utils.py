import warnings
from enum import Enum
from typing import Type, Iterable

from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QFile, QTextStream
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QGraphicsView, QGraphicsScene, QScrollArea, QApplication, \
    QStyleFactory
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from physiolabxr.configs import config
from physiolabxr.configs.config_ui import button_style_classic
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.exceptions.exceptions import RenaError
from physiolabxr.presets.presets_utils import get_all_preset_names, get_stream_preset_names
from physiolabxr.scripting.script_utils import validate_python_script_class
from physiolabxr.ui.dialogs import dialog_popup


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
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        elif position == "center":
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        elif position == "rightbottom":
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        elif position == "righttop":
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        elif position == "lefttop":
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)

        elif position == "leftbottom":
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
            ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignBottom)

        ql.setText(label)
        vl.addWidget(ql)

    return vl


def init_container(parent, label=None, label_position=None, label_bold=True, vertical=True, style=None, size=None,
                   insert_position=None):
    container = QtWidgets.QWidget()

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


def init_button(parent, label=None, function=None, style=button_style_classic):
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

def init_label(parent, text, max_width=None, max_hight=None, size=None):
    label = QLabel(text)

    parent.addWidget(label)
    if max_width:
        label.setMaximumWidth(max_width)
    if max_hight:
        label.setMaximumHeight(max_hight)

def init_scroll_label(parent, text, max_width=None, max_hight=None, size=None):
    label = ScrollLabel()
    label.setText(text=text)

    parent.addWidget(label)
    if max_width:
        label.setMaximumWidth(max_width)
    if max_hight:
        label.setMaximumHeight(max_hight)



def init_spec_view(parent, label, graph=None):
    if label:
        ql = QLabel()
        ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        ql.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        ql.setText(label)
        parent.addWidget(ql)

    spc_gv = QGraphicsView()
    parent.addWidget(spc_gv)

    scene = QGraphicsScene()
    spc_gv.setScene(scene)
    spc_gv.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    if graph:
        scene.addItem(graph)
    # spc_gv.setFixedSize(config.WINDOW_WIDTH/4, config.WINDOW_HEIGHT/4)
    return scene


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

def convert_numpy_to_uint8(array):
    return array.astype(np.uint8)


def stream_stylesheet(stylesheet_url):
    stylesheet = QFile(stylesheet_url)
    stylesheet.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
    stream = QTextStream(stylesheet)
    QApplication.instance().setStyleSheet(stream.readAll())

# def apply_stylesheet(widget):
#     theme = config.settings.value('theme')
#     widget.setStyleSheet(AppConfigs()._style_sheets[theme])

def add_presets_to_combobox(combobox: QComboBox):
    combobox.addItems([i for i in get_all_preset_names()])

def update_presets_to_combobox(combobox: QComboBox):  # TODO script should also call this when new preset is added
    combobox.clear()
    for i in get_all_preset_names():
        combobox.addItem(i)

def add_stream_presets_to_combobox(combobox):
    combobox.addItems([i for i in get_stream_preset_names()])

class AnotherWindow(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self, widget_to_add: QWidget, close_function: callable):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(widget_to_add)
        self.close_function = close_function
        self.setLayout(layout)

    def closeEvent(self, event):
        print('Window closed')
        if self.close_function():
            event.accept()  # let the window close
        else:
            event.ignore()


class another_window(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, window_title: str):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle(window_title)

    def get_layout(self):
        return self.layout


# class for scrollable label
class ScrollLabel(QScrollArea):

    # constructor
    def __init__(self, *args, **kwargs):
        QScrollArea.__init__(self, *args, **kwargs)

        # making widget resizable
        self.setWidgetResizable(True)

        # making qwidget object
        content = QWidget(self)
        self.setWidget(content)

        # vertical box layout
        lay = QVBoxLayout(content)

        # creating label
        self.label = QLabel(content)

        # setting alignment to the text
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        # making label multi-line
        self.label.setWordWrap(True)

        # adding label to the layout
        lay.addWidget(self.label)

    # the setText method
    def setText(self, text):
        # setting text to the label
        self.label.setText(text)

def clear_layout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()

def clear_widget(target_widget):
    for i in reversed(range(target_widget.count())):
        widget = target_widget.widget(i)
        widget.deleteLater()

def validate_script_path(script_path, desired_class: Type) -> bool:
    try:
        validate_python_script_class(script_path, desired_class)
    except RenaError as error:
        dialog_popup(str(error), title='Error')
        return False
    else:
        return True

def add_enum_values_to_combobox(combobox: QComboBox, enum: Type[Enum], can_be_selected_in_gui: list=None):
    """Add all the names of an enum to a combobox.
    If can_be_selected_in_gui is not None, only the values in the list are enabled.

    Notes:
        The strings that are added to the combobox are value of the enum, not the name.
        So if you need to call combobox.findText(...), the argument should be enum_var.name, not enum_var.value.

    """
    if can_be_selected_in_gui is None:
        combobox.addItems([name for name, member in enum.__members__.items()])
    else:
        for name, member in enum.__members__.items():
            combobox.addItem(name)
            item = combobox.model().item(combobox.count() - 1)
            if member not in can_be_selected_in_gui:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)

def show_label_movie(label: QLabel, is_show: bool):
    label.setVisible(is_show)
    if is_show:
        label.movie().start()
    else:
        label.movie().stop()

def get_int_from_line_edit(line_edit: QtWidgets.QLineEdit, name=""):
    """
    Get the int value from a line edit. Set the lineedit to red if the value is not an int.
    This function also connects the line edit to a function that will set the line edit back to normal when the text is changed.
    @param line_edit:
    @return:
    """
    try:
        return int(line_edit.text())
    except ValueError:
        line_edit.setStyleSheet("border: 2px solid red;")
        def set_back():
            line_edit.setStyleSheet("")
            line_edit.textChanged.disconnect(set_back)
        line_edit.textChanged.connect(set_back)
        raise RenaError(f'{name} must be an integer')
class ShortCutType(Enum):
    switch = 1
    delete = 2
    start = 3
    pop = 4

# def add_items(combobox: QComboBox, items: Iterable):
#     """
#     call addItems on a combobox
#     remove placeholder if there's any
#     """
#     placeholder_index = combobox.findText(AppConfigs()._placeholder_text)
#     if placeholder_index == -1:
#         warnings.warn(f"combobox {combobox.objectName()} has no placeholder, may subject to NSException.")
#     combobox.addItems(items)
#     # remove the placeholder item from the combobox if it exists
#     if placeholder_index != -1:
#         combobox.removeItem(placeholder_index)