from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QPixmap

from rena.config_ui import stream_widget_icon_size

start_stream_icon = QIcon('../media/icons/start.svg')
stop_stream_icon = QIcon('../media/icons/stop.svg')
options_icon = QIcon('../media/icons/options.svg')
pop_window_icon = QIcon('../media/icons/popwindow.svg')
dock_window_icon = QIcon('../media/icons/dockwindow.svg')
remove_stream_icon = QIcon('../media/icons/removestream.svg')

add_icon = QIcon('../media/icons/add.svg')
minus_icon = QIcon('../media/icons/minus.svg')



pause_icon = QIcon('../media/icons/pause.svg')

# Stream widget icon in the visualization tab
stream_unavailable_icon = QIcon('../media/icons/streamwidget_stream_unavailable.svg')
# stream_unavailable_pixmap = stream_unavailable_icon.pixmap(72, 72)
stream_available_icon = QIcon('../media/icons/streamwidget_stream_available.svg')
# stream_available_pixmap = stream_available_icon.pixmap(72, 72)
stream_active_icon = QIcon('../media/icons/streamwidget_stream_viz_active.svg')
# stream_active_pixmap = stream_active_icon.pixmap(72, 72)


# stream_unavailable_pixmap = QPixmap('../media/icons/streamwidget_stream_unavailable.png')
# stream_available_pixmap = QPixmap('../media/icons/streamwidget_stream_available.png')
# stream_active_pixmap = QPixmap('../media/icons/streamwidget_stream_viz_active.png')


# strings in Rena
# main window
num_active_streams_label_text = 'Streams: {0} added, {1} available, {2} streaming, {3} replaying'

# Recording tab
recording_tab_file_save_label_prefix = 'File will be saved as: '
start_recording_text = 'Start Recording'
stop_recording_text = 'Stop Recording'

# Scripting Widget
script_realtime_info_text = 'Loop (with overheads) per second {0}    Average loop call runtime {1}'

# StreamGroupView
CHANNEL_ITEM_IS_DISPLAY_CHANGED = 1
CHANNEL_ITEM_GROUP_CHANGED = 2
num_points_shown_text = 'Number of points shown: {0}'