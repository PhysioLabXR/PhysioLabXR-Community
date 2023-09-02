# start_stream_icon = load_media_icon('start.svg')
# stop_stream_icon = load_media_icon('stop.svg')
# terminate_icon = load_media_icon('terminate.svg')  # Only used in replay playback widget as of now
# options_icon = load_media_icon('options.svg')
# pop_window_icon = load_media_icon('pop_window.svg')
# dock_window_icon = load_media_icon('dock_window.svg')
# remove_stream_icon = load_media_icon('remove.svg')
#
# add_icon = load_media_icon('add.svg')
# minus_icon = load_media_icon('minus.svg')
#
# pause_icon = load_media_icon('pause.svg')
#
# collapse_icon = load_media_icon('collapse.svg')
# expand_icon = load_media_icon('expand.svg')

# Stream widget icon in the visualization tab
# stream_unavailable_icon = load_media_icon('streamwidget_stream_unavailable.svg')
# stream_unavailable_pixmap = stream_unavailable_icon.pixmap(72, 72)
# stream_available_icon = load_media_icon('streamwidget_stream_available.svg')
# stream_available_pixmap = stream_available_icon.pixmap(72, 72)
# stream_active_icon = load_media_icon('streamwidget_stream_viz_active.svg')
# stream_active_pixmap = stream_active_icon.pixmap(72, 72)


# stream_unavailable_pixmap = QPixmap('../_media/icons/streamwidget_stream_unavailable.png')
# stream_available_pixmap = QPixmap('../_media/icons/streamwidget_stream_available.png')
# stream_active_pixmap = QPixmap('../_media/icons/streamwidget_stream_viz_active.png')


# strings in Rena
# main window
num_active_streams_label_text = 'Streams: {0} added, {1} available, {2} streaming, {3} replaying'

# Recording tab
recording_tab_file_save_label_prefix = 'File will be saved as: '
recording_tab_file_save_label_prefix_csv = 'File will be saved as csv files in directory:'
start_recording_text = 'Start Recording'
stop_recording_text = 'Stop Recording'

# Scripting Widget
script_realtime_info_text = 'Loop (with overheads) per second {0}    Average loop call runtime {1}'

# Scripting Widget Tooltips
scripting_input_widget_shape_label_tooltip = 'The expected shape of this input data at every loop. \n' \
                                             'First dimension is the number of channels. \n' \
                                             'Second dimension is the number of time points (set by the input time window)'
scripting_input_widget_button_tooltip = "Remove this input"
scripting_input_widget_name_label_tooltip = "Name of the input stream"

# StreamGroupView
CHANNEL_ITEM_IS_DISPLAY_CHANGED = 1
CHANNEL_ITEM_GROUP_CHANGED = 2
num_points_shown_text = 'Number of points shown: {0}'