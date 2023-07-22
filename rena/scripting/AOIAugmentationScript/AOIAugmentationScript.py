import numpy as np

from rena.scripting.AOIAugmentationScript.AOIAugmentationUtils import generate_random_attention_matrix
from rena.scripting.RenaScript import RenaScript


class AOIAugmentationScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        default_attention_matrix = generate_random_attention_matrix(grid_shape=(25, 50))


    # Start will be called once when the run button is hit.
    def init(self):

        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # fixation_detection
        # list of gaze of intersection
        # show pixel on patch x
        # detected_fixation_on_display_area = [1000, 1000]

        # state machine







        print('Loop function is called')

    # cleanup is called when the stop button is hit



    def cleanup(self):
        print('Cleanup function is called')


