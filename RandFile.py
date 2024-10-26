from pylsl import local_clock
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.third_party.WearableSensing.DSI_py3 import *
import numpy as np
import sys
from physiolabxr.utils.buffers import DataBuffer
import sounddevice as sd
import time

class dummyscript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
    def init(self, arg = ''):
        self.frequency = 40
        self.duration = 10 * 60
        self.fs = 44100
        self.time = np.linspace(0, self.duration, int(self.fs*self.duration), endpoint = False)
        self.sinewave = 0.5 * np.sin(2 * np.pi * self.frequency * self.time)
        self.sound = 0
        time.sleep(15)
        self.sound = 1
        sd.play(self.sinewave, self.fs)

    def loop(self):
        self.set_output(stream_name='sound', data=[1], timestamp=local_clock())
        # if self.starttime == None:
        #     self.starttime = local_clock()
        # self.set_output(stream_name = 'sound', data = [0], timestamp  = local_clock())
        # if (local_clock() - self.starttime )> 1:
        #     self.set_output(stream_name='sound', data=[1], timestamp=local_clock())
    def cleanup(self):
        pass