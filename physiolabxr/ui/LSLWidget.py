# This Python file uses the following encoding: utf-8

from physiolabxr.exceptions.exceptions import ChannelMismatchError, UnsupportedErrorTypeError, LSLStreamNotFoundError
from physiolabxr.configs.configs import AppConfigs
from physiolabxr.presets.PresetEnums import PresetType, DataType
from physiolabxr.presets.Presets import Presets
from physiolabxr.presets.presets_utils import get_stream_preset_info
from physiolabxr.threadings import workers
from physiolabxr.ui.BaseStreamWidget import BaseStreamWidget
from physiolabxr.utils.ui_utils import dialog_popup


class LSLWidget(BaseStreamWidget):

    def __init__(self, parent_widget, parent_layout, stream_name, insert_position=None):
        """
        BaseStreamWidget is the main interface with plots and a single stream of data.
        @param parent_widget: the MainWindow class
        @param parent_layout: the layout of the parent widget, that is the layout of MainWindow's stream tab
        @param stream_name: the name of the stream
        """

        # GUI elements
        super().__init__(parent_widget, parent_layout, PresetType.LSL, stream_name,
                         data_timer_interval=AppConfigs().pull_data_interval, use_viz_buffer=True,
                         insert_position=insert_position)
        num_channels = get_stream_preset_info(self.stream_name, 'num_channels')
        lsl_worker = workers.LSLInletWorker(self.stream_name, num_channels, RenaTCPInterface=None)
        self.connect_worker(lsl_worker, True)
        self.connect_start_stop_btn(self.start_stop_stream_btn_clicked)
        self.start_timers()

    def start_stop_stream_btn_clicked(self):
        try:
            super().start_stop_stream_btn_clicked()
        except LSLStreamNotFoundError as e:
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
