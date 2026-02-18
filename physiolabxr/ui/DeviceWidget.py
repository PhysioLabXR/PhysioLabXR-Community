# This Python file uses the following encoding: utf-8
import traceback

from PyQt6.QtWidgets import QDialogButtonBox, QWidget

import physiolabxr.threadings.AudioWorkers
import physiolabxr.threadings.DeviceWorker
from physiolabxr.exceptions.exceptions import ChannelMismatchError, CustomDeviceStartStreamError, CustomDeviceStreamInterruptedError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.presets.presets_utils import get_stream_preset_info
from physiolabxr.threadings import workers
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
from physiolabxr.ui.DeviceMessageConsole import DeviceMessageConsole
from physiolabxr.ui.dialogs import dialog_popup
from physiolabxr.utils.ui_utils import show_label_movie


class DeviceWidget(BaseStreamWidget):

    def __init__(self, parent_widget, parent_layout, stream_name, insert_position=None):
        """
        BaseStreamWidget is the main interface with plots and a single stream of data.
        Args:
            parent_widget: the MainWindow class
            parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
            stream_name: the name of the stream

        How to add a custom device options UI to a custom device:
        """

        # GUI elements
        super().__init__(parent_widget, parent_layout, PresetType.CUSTOM, stream_name,
                         data_timer_interval=AppConfigs().pull_data_interval, use_viz_buffer=True,
                         insert_position=insert_position)

        self.device_msg_console_btn.setIcon(AppConfigs()._icon_terminal)
        self.device_options_btn.setIcon(AppConfigs()._icon_device_options)

        device_worker = physiolabxr.threadings.DeviceWorker.DeviceWorker(self.stream_name)
        self.connect_worker(device_worker, False)  # this will register device worker as self.data_worker
        if device_worker.device_options_widget_class is not None:
            self.register_device_options_widgets(device_worker.device_options_widget_class)
        else:
            self.device_options_widget = None

        self.device_msg_console = DeviceMessageConsole(self, self.data_worker)
        self.device_msg_console_btn.clicked.connect(self.show_focus_device_msg_console_widget)
        self.device_msg_console_btn.show()
        self.show_focus_device_msg_console_widget()

        self.start_timers()
        self.first_frame_received = False

    def start_stop_stream_btn_clicked(self):
        if not self.is_streaming():
            self.first_frame_received = False
            show_label_movie(self.waiting_label, True)
        try:
            # this segment is the same as in the BaseStreamWidget, except the start_stream include the start_stream_arguments
            # from the device options widget
            if self.data_worker.is_streaming:
                self.data_worker.stop_stream()
                if not self.data_worker.is_streaming and self.add_stream_availability:
                    self.update_stream_availability(self.data_worker.is_stream_available())
            else:
                try:
                    start_stream_args = self.device_options_widget.start_stream_args() if self.device_options_widget is not None else {}
                except Exception as e:
                    traceback.print_exc()
                    raise Exception(f"Error in getting start stream arguments from device options widget: {e}")
                self.data_worker.start_stream(**start_stream_args)
            self.set_button_icons()
            self.main_parent.update_active_streams()
        except CustomDeviceStartStreamError as e:
            show_label_movie(self.waiting_label, False)  # Stop loading animation
            self.set_button_icons()  # Reset button state
            self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
            return
        except CustomDeviceStreamInterruptedError as e:
            show_label_movie(self.waiting_label, False)
            self.set_button_icons()
            self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
            return
        except ChannelMismatchError as e:  # only LSL's channel mismatch can be checked at this time, zmq's channel mismatch can only be checked when receiving data
            preset_chan_num = get_stream_preset_info(self.stream_name, 'num_channels')
            message = f'The stream with name {self.stream_name} found on the network has {e.message}.\n The preset has {preset_chan_num} channels. \n Do you want to reset your preset to a default and start stream.\n You can edit your stream channels in Options if you choose Cancel'
            reply = dialog_popup(msg=message, title='Channel Mismatch', mode='modal', main_parent=self.main_parent,
                                 buttons=self.channel_mismatch_buttons)
            if reply.result():
                self.reset_preset_by_num_channels(e.message, get_stream_preset_info(self.stream_name, 'data_type'))
                try:
                    self.data_worker.start_stream()  # start the stream again with updated preset
                    self.set_button_icons()
                    self.main_parent.update_active_streams()
                except LSLStreamNotFoundError as e:
                    self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
                    return
            else:
                return
        except Exception as e:
            raise UnsupportedErrorTypeError(str(e))

    def process_stream_data(self, data_dict):
        """

        If the data_dict contains an error message, a dialog popup will be shown and the stream will be stopped.
        """
        # check if there is an error decoding the zmq message
        if 'e' in data_dict:
            dialog_popup(msg=f"{self.stream_name} streaming interrupted \n {data_dict['e']}", title='Error', mode='modal', main_parent=self.main_parent, buttons=QDialogButtonBox.StandardButton.Ok)
            show_label_movie(self.waiting_label, False)
            self.start_stop_stream_btn_clicked()  # stop streaming
            return
        if not self.first_frame_received and 'frames' in data_dict and len(data_dict['frames']) > 0:
            self.first_frame_received = True
            show_label_movie(self.waiting_label, False)
        super().process_stream_data(data_dict)

    def show_focus_device_msg_console_widget(self):
        self.device_msg_console.show()
        self.device_msg_console.activateWindow()

    def register_device_options_widgets(self, device_options_widget_class):
        """Register the device options widget to the device widget

        This function is called in the device worker if the function create_custom_device_classes returns a device_options_widget.
        """
        self.device_options_widget: QWidget = device_options_widget_class(self.stream_name, self.data_worker.device_interface)
        def show_focus_device_options_widget():
            self.device_options_widget.show()
            self.device_options_widget.activateWindow()
        self.device_options_btn.clicked.connect(show_focus_device_options_widget)
        self.device_options_btn.show()
        show_focus_device_options_widget()

    def remove_stream(self):
        super().remove_stream()
        if self.device_options_widget is not None:
            self.device_options_widget.close()
        self.device_msg_console.close()