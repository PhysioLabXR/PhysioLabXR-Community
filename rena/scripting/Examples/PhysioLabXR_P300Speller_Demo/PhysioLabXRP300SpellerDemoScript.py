import numpy as np

from rena.scripting.RenaScript import RenaScript
import pickle
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet
from sklearn.linear_model import LogisticRegression
from rena.utils.buffers import DataBuffer


class PhysioLabXRP300SpellerDemoScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        # initialize model
        self.model = LogisticRegression()


    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        print('Loop function is called')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
