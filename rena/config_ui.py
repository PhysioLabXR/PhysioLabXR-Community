color_pink = '#fff5f6'
color_white = '#ffffff'
color_gray = '#f0f0f0'
color_bright = '#E8E9EB'
color_green = "#03fc4e"
color_red = '#d1000e'

button_color = color_white

button_style_classic = "background-color: " + button_color + "; border-style: outset; border-width: 2px; " \
                                                             "border-radius: 10px; " \
                                                             "border-color: gray; font: 12px; min-width: 10em; " \
                                                             "padding: 6px; "

default_theme = 'dark'

inference_button_style = "max_width: 250px;, min_width: 100px; font: bold 14px"
sensors_type_ui_name_dict = {'OpenBCICyton': 'OpenBCI Cyton',
                             }

sensor_ui_name_type_dict = {v: k for k, v in sensors_type_ui_name_dict.items()}

default_add_lsl_data_type = 'YourStreamName'

sampling_rate_decimal_places = 2
visualization_fps_decimal_places = 2
tick_frequency_decimal_places = 2

# cam_display_width = 640
# cam_display_height = 480

capture_display_width = 2560
capture_display_height = 1440

nothing_selected = 0
channel_selected = 1
channels_selected = 2
group_selected = 3
groups_selected = 4
mix_selected = 5

stream_widget_icon_size = [72, 72]

image_min_width = 512
image_min_height = 512

new_group_default_plot_format = 1