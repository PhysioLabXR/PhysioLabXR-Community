from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.WearableSensing.DSI_py3 import *
import numpy as np
import sys


@MessageCallback
def NullMessageCallback(msg, lvl=0):
    return 1


@SampleCallback
def NullSampleCallback(headsetPtr, packetTime, userData):
    pass


@MessageCallback
def ExampleMessageCallback(msg, lvl=0):
    if lvl <= 3:  # ignore messages at debugging levels higher than 3
        print("DSI Message (level %d): %s" % (lvl, IfStringThenNormalString(msg)))
    return 1

@SampleCallback
def ExampleSampleCallback_Signals(headsetPtr, packetTime, userData, data, time ):
    h = Headset(headsetPtr)
    # strings = ['%s=%+08.2f' % (IfStringThenNormalString(ch.GetName()), ch.ReadBuffered()) for ch in h.Channels()]
    values = [ch.ReadBuffered() for ch in h.Channels()]
    data = np.append(data, values, axis = 0)
    time = np.append(time, values, axis = 0)
    # sys.stdout.flush()

@SampleCallback
def ExampleSampleCallback_Impedances(headsetPtr, packetTime, userData):
    h = Headset(headsetPtr)
    fmt = '%s = %5.3f'
    strings = [fmt % (IfStringThenNormalString(src.GetName()), src.GetImpedanceEEG()) for src in h.Sources() if
               src.IsReferentialEEG() and not src.IsFactoryReference()]
    strings.append(fmt % ('CMF @ ' + h.GetFactoryReferenceString(), h.GetImpedanceCMF()))
    print(('%8.3f:   ' % packetTime) + ', '.join(strings))
    sys.stdout.flush()

class DSI_24(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def ExampleSampleCallback_Signals(self, headsetPtr, packetTime, userData,):
        h = Headset(headsetPtr)
        # strings = ['%s=%+08.2f' % (IfStringThenNormalString(ch.GetName()), ch.ReadBuffered()) for ch in h.Channels()]
        values = [ch.ReadBuffered() for ch in h.Channels()]
        self.h.data = np.append(self.h.data, values, axis=0)
        self.h.time = np.append(self.h.time, values, axis=0)
    def init(self, port, arg = ''):
        #Setting H to th headset class
        self.h = Headset()
        #Connecting the headset class to the correct COM port
        self.h.Connect('COM3')
        self.h.data = np.array([])
        self.h.time = np.array([])
        self.h.SetMessageCallback(self.ExampleMessageCallback)  # could set this to NullMessageCallback instead if we wanted to shut it up
        if arg.lower().startswith('imp'):
            self.h.SetSampleCallback(self.ExampleSampleCallback_Impedances, 0)
            self.h.StartImpedanceDriver()
        else:
            self.h.SetSampleCallback(self.ExampleSampleCallback_Signals(self.h.data, self.h.time), 0)
            if len(arg.strip()):
                self.h.SetDefaultReference(arg, True)
        args = getattr(sys, 'argv', [''])
        if sys.platform.lower().startswith('win'):
            default_port = 'COM3'
        else:
            default_port = '/dev/cu.DSI7-0009.BluetoothSeri'

        # first command-line argument: serial port address
        if len(args) > 1:
            port = args[1]
        else:
            port = default_port

        # second command-line argument:  name of the Source to be used as reference, or the word 'impedances'
        if len(args) > 2:
            ref = args[2]
        else:
            ref = ''
        #Start the data Acquisition
        self.h.StartDataAcquisition()
    def loop(self):
        'sample call back'
        self.set_output(stream_name = "WearableSensing", data=self.h.data, time = self.h.time) #Will need to change time to reflext the time the data acquisition started
        self.h.data = np.array([])
        self.h.time = np.array([])

    def cleanup(self):
        self.h.StopDataAcquisition()

