from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.thirdparty.WearableSensing.DSI_py3 import *
import numpy as np
import sys
from physiolabxr.utils.buffers import DataBuffer
from physiolabxr.utils.time_utils import get_clock_time

#Creating a data buffer with the DataBuffer class
data_buffer = DataBuffer()
lsl_offset_time = None

@MessageCallback
def ExampleMessageCallback(msg, lvl=0):
    if lvl <= 3:  # ignore messages at debugging levels higher than 3
        print("DSI Message (level %d): %s" % (lvl, IfStringThenNormalString(msg)))
    return 1


@SampleCallback
def ExampleSampleCallback_Signals(headsetPtr, packetTime, userData):
    #This is the function that will be called every time a new packet is received
    global data_buffer
    global lsl_offset_time
    #Grab the headset by using a pointer
    h = Headset(headsetPtr)
    #Get the signal from each channel and format it so that it can be created into an array
    new_data = np.array(['%+08.2f' % (ch.GetSignal()) for ch in h.Channels()])
    #Reshapes the array into a 24x1 array so that it can be inputted into the data_buffer
    new_data = new_data.reshape(24,1)
    #Rearrange new_data to fit with desired output format
    new_data = new_data[[2,3,4,1,7,5,0,6],:] #Changed from normal sensing script to only have F3, Fz, F4, C3, Cz, C4, P3, P4
    #Get the time of the packet as a temporary solution to timestamps
    if lsl_offset_time is None:
        lsl_offset_time = get_clock_time() - float(packetTime)

    t = [float(packetTime) + lsl_offset_time]
    #Create a dictionary with the stream name, data, and timestamps
    new_data_dict = {
        'stream_name': 'DSI-24',
        'frames': new_data,
        'timestamps': t
    }
    #Update the data buffer with the new data
    data_buffer.update_buffer(new_data_dict)


@SampleCallback
def ExampleSampleCallback_Impedances(headsetPtr, packetTime, userData):
    #Not yet used
    h = Headset(headsetPtr)
    fmt = '%s = %5.3f'
    strings = [fmt % (IfStringThenNormalString(src.GetName()), src.GetImpedanceEEG()) for src in h.Sources() if
               src.IsReferentialEEG() and not src.IsFactoryReference()]
    strings.append(fmt % ('CMF @ ' + h.GetFactoryReferenceString(), h.GetImpedanceCMF()))
    print(('%8.3f:   ' % packetTime) + ', '.join(strings))
    sys.stdout.flush()

class DSI24(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    def init(self, arg = ''):
        #Create a headset object
        self.headset = Headset()
        #Set the message callback to ExampleMessageCallback
        self.headset.SetMessageCallback(ExampleMessageCallback)
        #Retrieves the command line arguments
        args = getattr(sys, 'argv', [''])
        #Set the default port to the first command line argument based on the parameter provided by user
        default_port = self.params['COM Port']
        #Connect the headset
        self.headset.Connect(default_port)
        #Start the data acquisition based on the parameter provided by user
        if arg.lower().startswith('imp'):
            #Currently not used
            self.headset.SetSampleCallback(ExampleSampleCallback_Impedances, 0)
            self.headset.StartImpedanceDriver()
        else:
            #Set the sample callback to ExampleSampleCallback_Signals
            self.headset.SetSampleCallback(ExampleSampleCallback_Signals, 0)
            if len(arg.strip()): self.headset.SetDefaultReference(arg, True)
        #Start the data acquisition
        self.headset.StartBackgroundAcquisition()

    def loop(self):
        #Called every loop based on the user's chosen frequency
        global data_buffer
        #If the data buffer has data, then set the output to the data buffer
        if len(data_buffer.keys()) > 0:
            self.set_output(stream_name = 'OpenBCICyton8Channels', data = data_buffer.get_data('DSI-24'), timestamp = data_buffer.get_timestamps('DSI-24'))
            #Clear the data buffer
            data_buffer.clear_stream_buffer_data('DSI-24')

    def cleanup(self):
        #Called when the script is stopped
        global data_buffer
        #Stop the data acquisition
        self.headset.StopDataAcquisition()
        #Disconnect the headset
        data_buffer.clear_buffer()