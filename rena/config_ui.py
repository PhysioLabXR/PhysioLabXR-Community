color_pink = '#fff5f6'
color_white = '#ffffff'
color_gray = '#f0f0f0'
color_bright = '#E8E9EB'

button_color = color_white

button_style_classic = "background-color: " + button_color + "; border-style: outset; border-width: 2px; " \
                                                             "border-radius: 10px; " \
                                                             "border-color: gray; font: 12px; min-width: 10em; " \
                                                             "padding: 6px; "

default_theme = 'dark'

inference_button_style = "max_width: 250px;, min_width: 100px; font: bold 14px"
sensors_type_ui_name_dict = {'OpenBCICyton': 'OpenBCI Cyton',
                             # 'RNUnityEyeLSL': 'Vive Pro eye-tracking (Unity)',
                             }

sensor_ui_name_type_dict = {v: k for k, v in sensors_type_ui_name_dict.items()}


default_add_lsl_data_type = 'YourStreamName'

sampling_rate_decimal_places = 2

cam_display_width = 640
cam_display_height = 480

capture_display_width = 640
capture_display_height = 480