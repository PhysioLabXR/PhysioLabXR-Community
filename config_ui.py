color_pink = '#fff5f6'
color_white = '#ffffff'
color_gray = '#f0f0f0'
color_bright = '#E8E9EB'

button_color = color_white

button_style_classic = "background-color: " + button_color + "; border-style: outset; border-width: 2px; " \
                                                             "border-radius: 10px; " \
                                                             "border-color: gray; font: 12px; min-width: 10em; " \
                                                             "padding: 6px; "
sensors_type_ui_name_dict = {'OpenBCICyton': 'OpenBCI Cyton',
                             # 'RNUnityEyeLSL': 'Vive Pro eye-tracking (Unity)',
                             }

sensor_ui_name_type_dict = {v: k for k, v in sensors_type_ui_name_dict.items()}


default_add_lsl_data_type = 'Unity.EventMarkers'

sampling_rate_decimal_places = 2

cam_disply_width = 640
cam_display_height = 480