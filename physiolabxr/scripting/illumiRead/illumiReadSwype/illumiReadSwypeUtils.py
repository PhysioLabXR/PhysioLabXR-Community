import numpy as np

from physiolabxr.scripting.illumiRead.illumiReadSwype import illumiReadSwypeConfig
from physiolabxr.scripting.illumiRead.illumiReadSwype.illumiReadSwypeConfig import KeyIndexIDDict


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
        

def word_candidate_list_to_lvt(words_list, max_length=1024):
    """
    Convert a words sequence to a list of integers representing the characters
    """
    lvt = [0]
    lvt.append(len(words_list))
    overflow_flag = False

    for word_index, word in enumerate(words_list):
        word_lvt = []
        word_lvt.append(word_index)
        word_lvt.append(len(word))
        for char in word:
            chat_int = KeyIndexIDDict[char.upper()]
            word_lvt.append(chat_int)

        lvt += word_lvt

        if len(lvt) >= max_length:
            overflow_flag = True
            break

    if not overflow_flag:
        lvt += [0] * (max_length - len(lvt))
    else:
        lvt[0] = 1

    return lvt, overflow_flag


if __name__ == '__main__':

    word_candidate_list_to_lvt(['hello', 'world', 'this'])
