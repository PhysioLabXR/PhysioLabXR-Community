import numpy as np
import pickle
import sys
import getopt
import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock

file_path = 'TobiiProFusionUnityLSLOutlet_01.pickle'
# Open the file in binary mode
with open(file_path, 'rb') as file:
    # Load the data from the file
    gaze_raw = pickle.load(file)

gaze_data = gaze_raw[0]
timestamps = gaze_raw[1]

print(gaze_data.shape)


"""Example program to demonstrate how to send a multi-channel time series to
LSL."""



def main(argv):
    srate = 250
    name = 'TobiiProFusionUnityLSL'
    print('Stream name is ' + name)
    type = 'EEG'
    n_channels = gaze_data.shape[0]
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    try:
        opts, args = getopt.getopt(argv, "hs:c:n:t:", longopts=["srate=", "channels=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--srate"):
            srate = float(arg)
        elif opt in ("-c", "--channels"):
            n_channels = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            type = arg
    info = StreamInfo(name, type, n_channels, srate, 'float32', 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0
    gaze_data_index = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            # mysample = [rand()*10 for _ in range(n_channels)]
            # # now send it
            # outlet.push_sample(mysample)
            print(gaze_data_index)
            outlet.push_sample(gaze_data[:, gaze_data_index])
            # print("pushed sample")
            gaze_data_index += 1
            if gaze_data_index >= gaze_data.shape[1]:
                gaze_data_index = 0



        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)


if __name__ == '__main__':
    main(sys.argv[1:])
