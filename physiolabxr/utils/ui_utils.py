from enum import Enum
from typing import Type

from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtCore import QFile, QTextStream
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QDialog, QDialogButtonBox, \
    QGraphicsView, QGraphicsScene, QCheckBox, QScrollArea, QApplication
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from physiolabxr.configs.config import settings
from physiolabxr.configs.config_ui import button_style_classic
from physiolabxr.exceptions.exceptions import RenaError
from physiolabxr.presets.presets_utils import get_all_preset_names, get_stream_preset_names
from physiolabxr.scripting.script_utils import validate_python_script_class


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



# def init_camera_widget(parent, label_string, insert_position):
#     container_widget, layout = init_container(parent=parent, insert_position=insert_position)
#     container_widget.setFixedWidth(120 + max(config_ui.cam_display_width, config_ui.capture_display_width))
#
#     camera_img_label = QLabel()
#     _, label_btn_layout = init_container(parent=layout, vertical=False)
#     cam_id_label = QLabel(label_string)
#     cam_id_label.setStyleSheet("font: bold 14px;")
#     label_btn_layout.addWidget(cam_id_label)
#     remove_cam_btn = init_button(parent=label_btn_layout, label='Remove Capture')
#     layout.addWidget(camera_img_label)
#
#     return container_widget, layout, remove_cam_btn, camera_img_label


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


# def init_sensor_or_lsl_widget(parent, label_string, insert_position):
#     container_widget, layout = init_container(parent=parent, insert_position=insert_position)
#
#     _, top_layout = init_container(parent=layout, vertical=False)
#     ql = QLabel(config_ui.sensors_type_ui_name_dict[
#                     label_string] if label_string in config_ui.sensors_type_ui_name_dict.keys() else label_string)
#     ql.setStyleSheet("font: bold 14px;")
#     top_layout.addWidget(ql)
#
#     pop_window_btn = init_button(parent=top_layout, label=None)
#     remove_stream_btn = init_button(parent=layout, label=None)
#     signal_settings_btn = init_button(parent=layout, label=None)
#     start_stop_stream_btn = init_button(parent=layout, label=None)
#
#     # signal_settings_btn.setFixedWidth(200)
#     # pop_window_btn.setFixedWidth(200)
#     # remove_stream_btn.setFixedWidth(200)
#     # start_stop_stream_btn.setFixedWidth(200)
#     # start_stop_stream_btn.setIcon(QIcon('../_media/icons/stop.svg'))
#     start_stop_stream_btn.setText("Start Stream")
#     remove_stream_btn.setText("Remove Stream")
#     pop_window_btn.setText("Pop Window")
#     signal_settings_btn.setText("Signal")
#
#     return container_widget, layout, start_stop_stream_btn, pop_window_btn, signal_settings_btn, remove_stream_btn

# def init_sensor_or_lsl_widget(parent, label_string, insert_position):
#     container_widget, layout = init_container(parent=parent, insert_position=insert_position)
#
#     _, top_layout = init_container(parent=layout, vertical=False)
#     ql = QLabel(config_ui.sensors_type_ui_name_dict[
#                     label_string] if label_string in config_ui.sensors_type_ui_name_dict.keys() else label_string)
#     ql.setStyleSheet("font: bold 14px;")
#     signal_settings_btn = init_button(parent=top_layout, label='Signal Settings')
#     pop_window_btn = init_button(parent=top_layout, label='Pop Window')
#
#     signal_settings_btn.setFixedWidth(200)
#     pop_window_btn.setFixedWidth(200)
#
#     top_layout.addWidget(ql)
#     top_layout.addWidget(signal_settings_btn)
#     top_layout.addWidget(pop_window_btn)
#
#     start_stop_stream_btn = init_button(parent=layout, label='Start Stream')
#     # stop_stream_btn = init_button(parent=layout, label='Stop Stream')
#     # stop_stream_btn.setIcon(QIcon('../_media/icons/random.png'))
#     return container_widget, layout, start_stop_stream_btn, pop_window_btn, signal_settings_btn

# def init_add_widget(parent):
#     container, layout = init_container(parent=parent, label='Add Stream', label_bold=True)
#     container.setFixedWidth(700)
#
#     container_add_camera, layout_add_camera = init_container(parent=layout,
#                                                              label='Select a Camera(ID) or Screen Capture to add',
#                                                              vertical=False)
#     # detect camera
#     cameras = get_working_camera_ports()
#     cameras = list(map(str, cameras))
#     camera_screen_cap_list = cameras + ['monitor1']
#     # add camera container
#     camera_combo_box = init_combo_box(parent=layout_add_camera, label=None,
#                                       item_list=camera_screen_cap_list)
#     add_camera_btn = init_button(parent=layout_add_camera, label='Add')
#
#     reload_presets_btn = init_button(parent=layout, label='Reload Stream/Device/Experiment _presets')
#
#     # add LSL UI elements ######################
#     container_add_stream, layout_add_stream = init_container(parent=layout, label='Select a Stream to Add',
#                                                              vertical=False)
#     config.settings.beginGroup('presets/lslpresets')
#     stream_combo_box = init_combo_box(parent=layout_add_stream, label=None,
#                                       item_list=list(config.settings.childGroups()))
#     config.settings.endGroup()
#     add_stream_btn = init_button(parent=layout_add_stream, label='Add LSL')
#
#     # add device UI elements ######################
#     container_connect_device, layout_connect_device = init_container(parent=layout, label='Select a Device to Connect',
#                                                                      vertical=False)
#     config.settings.beginGroup('presets/devicepresets')
#     device_combo_box = init_combo_box(parent=layout_connect_device, label=None,
#                                       item_list=list(config.settings.childGroups()))
#     config.settings.endGroup()
#
#     add_device_btn = init_button(parent=layout_connect_device, label='Add Device')
#     # add experiment UI elements ######################
#     container_experiment, layout_experiment = init_container(parent=layout, label='Select an Experiment Preset to Start',
#                                                              vertical=False)
#     config.settings.beginGroup('presets/experimentpresets')
#     experiment_combo_box = init_combo_box(parent=layout_experiment, label=None,
#                                           item_list=list(config.settings.childGroups()))
#     config.settings.endGroup()
#
#     add_experiment_btn = init_button(parent=layout_experiment, label='Connect Experiment Streams/Devices')
#
#     container_add_lsl, layout_add_lsl = init_container(parent=layout, label='Define a Stream to Add', vertical=False)
#     _, lsl_data_type_input = init_inputBox(parent=layout_add_lsl, default_input=config_ui.default_add_lsl_data_type)
#     add_lsl_btn = init_button(parent=layout_add_lsl, label='Add')
#
#     return layout, camera_combo_box, add_camera_btn, stream_combo_box, add_stream_btn, lsl_data_type_input, add_lsl_btn, \
#            reload_presets_btn, device_combo_box, add_device_btn,experiment_combo_box,add_experiment_btn


class CustomDialog(QDialog):
    def __init__(self, title, msg, dialog_name, enable_dont_show, parent=None, buttons=QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel):
        super().__init__(parent=parent)

        self.setWindowTitle(title)
        self.dialog_name = dialog_name

        QBtn = buttons

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.msg = msg

        self.layout = QVBoxLayout()
        message = QLabel(str(msg))

        # center message and button
        self.layout.addWidget(message, alignment=Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.buttonBox, alignment=Qt.AlignmentFlag.AlignCenter)

        if enable_dont_show:
            # self.dont_show_button = QPushButton()
            self.dont_show_button = QCheckBox("Don't show this again")
            self.layout.addWidget(self.dont_show_button)
            self.dont_show_button.stateChanged.connect(self.toggle_dont_show)

        self.setLayout(self.layout)

    def toggle_dont_show(self):
        if self.dont_show_button.isChecked():
            settings.setValue('show_' + self.dialog_name, False)
            print('will NOT show ' + self.dialog_name)
        else:
            settings.setValue('show_' + self.dialog_name, True)
            print('will show ' + self.dialog_name)


def dialog_popup(msg, mode='modal', title='Warning', dialog_name=None, enable_dont_show=False, main_parent=None, buttons=QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel):
    if enable_dont_show:
        try:
            assert dialog_name is not None
        except AssertionError:
            print("dev: to use enable_dont_show, the dialog must have a unique identifier. Add the identifier by giving"
                  "the dialog_name parameter")
            raise AttributeError
        if settings.contains('show_' + dialog_name) and settings.value('show_' + dialog_name) == 'false':
            print('Skipping showing dialog ' + dialog_name)
            return
    dlg = CustomDialog(title, msg, dialog_name, enable_dont_show, buttons=buttons)  # If you pass self, the dialog will be centered over the main window as before.
    if main_parent:
        main_parent.current_dialog = dlg
    if mode=='modal':
        dlg.activateWindow()
        if dlg.exec():
            print("Dialog popup")
        else:
            print("Dialog closed")
    elif mode=='modeless':
        print("Showing modeless dialog")
        dlg.show()
        dlg.activateWindow()
    else:
        raise NotImplementedError
    return dlg


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


# def convert_rgb_to_qt_image(rgb_image, scaling_factor=1):
#     """Convert from an opencv image to QPixmap"""
#     h, w, ch = rgb_image.shape
#     bytes_per_line = ch * w
#     rgb_image = rgb_image.copy()
#     q_image = QtGui.QImage(rgb_image, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
#     q_pixelmap = QPixmap.fromImage(q_image)
#     q_pixelmap = q_pixelmap.scaled(
#                                 scaling_factor*w,
#                                 scaling_factor*h,
#                                 pg.QtCore.Qt.KeepAspectRatio) # rescale it
#     return q_pixelmap

# def convert_array_to_qt_heatmap(spec_array, scaling_factor):
#     h, w = spec_array.shape
#     heatmap_qim = array_to_colormap_qim(spec_array, normalize=True)
#     qpixmap = QPixmap(heatmap_qim)
#     qpixmap = qpixmap.scaled(scaling_factor*w, scaling_factor*h, pg.QtCore.Qt.KeepAspectRatio)  # resize spectrogram
#     return qpixmap

# def array_to_colormap_qim(a, normalize=True):
#     im = plt.imshow(a)
#     color_matrix = im.cmap(im.norm(im.get_array()))
#     qim = qimage2ndarray.array2qimage(color_matrix, normalize=normalize)
#     return qim

def stream_stylesheet(stylesheet_url):
    stylesheet = QFile(stylesheet_url)
    stylesheet.open(QFile.OpenModeFlag.ReadOnly | QFile.OpenModeFlag.Text)
    stream = QTextStream(stylesheet)
    QApplication.instance().setStyleSheet(stream.readAll())

def add_presets_to_combobox(combobox: QComboBox):
    for i in get_all_preset_names():
        combobox.addItem(i)

def update_presets_to_combobox(combobox: QComboBox):  # TODO script should also call this when new preset is added
    combobox.clear()
    for i in get_all_preset_names():
        combobox.addItem(i)

def add_stream_presets_to_combobox(combobox):
    for i in get_stream_preset_names():
        combobox.addItem(i)

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

def add_enum_values_to_combobox(combobox: QComboBox, enum: Type[Enum]):
    combobox.addItems([name for name, member in enum.__members__.items()])


def show_label_movie(label: QLabel, is_show: bool):
    label.setVisible(is_show)
    if is_show:
        label.movie().start()
    else:
        label.movie().stop()

