# This Python file uses the following encoding: utf-8
from PyQt6.QtWidgets import QDialogButtonBox

import physiolabxr.threadings.AudioWorkers
import physiolabxr.threadings.DeviceWorker
from physiolabxr.exceptions.exceptions import ChannelMismatchError, CustomDeviceStartStreamError, CustomDeviceStreamInterruptedError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType
from physiolabxr.presets.presets_utils import get_stream_preset_info
from physiolabxr.threadings import workers
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
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
        self.DeviceBtn.show()  # the device button is only shown for custom devices
        device_worker = physiolabxr.threadings.DeviceWorker.DeviceWorker(self.stream_name, device_widget=self)
        device_worker.device_widget = self
        self.connect_worker(device_worker, False)
        self.start_timers()
        self.first_frame_received = False

    def start_stop_stream_btn_clicked(self):
        if not self.is_streaming():
            self.first_frame_received = False
            show_label_movie(self.waiting_label, True)
        try:
            super().start_stop_stream_btn_clicked()
        except CustomDeviceStartStreamError as e:
            self.main_parent.current_dialog = dialog_popup(msg=str(e), title='ERROR')
            return
        except CustomDeviceStreamInterruptedError as e:
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
