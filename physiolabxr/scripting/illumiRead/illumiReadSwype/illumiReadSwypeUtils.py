import numpy as np

from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig


class illumiReadSwypeUserInput:
    def __init__(self, user_input_data_t, timestamp):

        self.gaze_hit_keyboard_background = user_input_data_t[
            illumiReadSwypeConfig.UserInputLSLStreamInfo.GazeHitKeyboardBackgroundChannelIndex]

        self.keyboard_background_hit_point_local = [
            user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalXChannelIndex],
            user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalYChannelIndex],
            user_input_data_t[
                illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyboardBackgroundHitPointLocalZChannelIndex]
        ]

        self.gaze_hit_key = user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.GazeHitKeyChannelIndex]

        self.key_hit_point_local = [
            user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalXChannelIndex],
            user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalYChannelIndex],
            user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitPointLocalZChannelIndex]
        ]

        self.key_hit_index = user_input_data_t[illumiReadSwypeConfig.UserInputLSLStreamInfo.KeyHitIndexChannelIndex]

        self.user_input_button_1 = user_input_data_t[
            illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton1ChannelIndex]  # swyping invoker

        self.user_input_button_2 = user_input_data_t[
            illumiReadSwypeConfig.UserInputLSLStreamInfo.UserInputButton2ChannelIndex]

        self.timestamp = timestamp