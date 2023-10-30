"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import random
import sys
import getopt
import string
import numpy as np
import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock
from scipy.special import softmax
import keyboard

target_locked = False

def on_key_event(event):
    global target_locked
    if event.event_type == keyboard.KEY_DOWN:
        print(f"Key {event.name} was pressed")
        if event.name == 'q':
            target_locked = True
            print('Target locked')
        if event.name == 'w':
            target_locked = False
            print('Target unlocked')



def main(argv):
    keyboard.hook(on_key_event)
    srate = 3
    name = 'Target vs Distractor Prediction'
    print('Stream name is ' + name)
    type = 'EEG'
    n_channels = 2
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
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            if target_locked==False:
                prediction_result = [random.random(), random.random()]
            else:
                prediction_result = [1+0.2*random.random(), 0.2*random.random()]
            prediction_result = softmax(prediction_result)
            mysample = prediction_result
            # now send it
            outlet.push_sample(mysample)
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)


if __name__ == '__main__':
    main(sys.argv[1:])
