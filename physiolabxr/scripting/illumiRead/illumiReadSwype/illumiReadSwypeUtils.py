import numpy as np

class illumiReadSwypeUserInput():
    def __init__(self, gaze_hit_keyboard_background, keyboard_background_hit_point_local, gaze_hit_key, key_hit_point_local, key_hit_index, user_input_button_1, user_input_button_2, timestamp):
        self.gaze_hit_keyboard_background = gaze_hit_keyboard_background
        self.keyboard_background_hit_point_local = keyboard_background_hit_point_local
        self.gaze_hit_key = gaze_hit_key
        self.key_hit_point_local = key_hit_point_local
        self.key_hit_index = key_hit_index
        self.user_input_button_1 = user_input_button_1
        self.user_input_button_2 = user_input_button_2
        self.timestamp = timestamp



