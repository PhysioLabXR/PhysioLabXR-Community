import scipy.signal
import sys
import getopt
import numpy as np
import time

from pylsl import StreamInfo, StreamOutlet, local_clock



# artifact_freq=0.1, artifact_amplitute = 1

def generate_dummy_eeg_signal(channel_num, fs, duration, add_level=False):
    eeg = []
    for i in range(channel_num):
        t = np.arange(0, duration, 1 / fs)
        noise = np.random.normal(0, 1, len(t))
        b, a = scipy.signal.butter(4, [4, 50], btype='bandpass', fs=fs)
        signal = scipy.signal.filtfilt(b, a, noise)
        eeg.append(signal)

    # if add_level:
    #     eeg = np.array(eeg)
    #     eeg = eeg + np.linspace(0, 100, channel_num).reshape(-1, 1)
    #     eeg = eeg.tolist()

    # if add_motion_artifact:
    #
    #     for channel_index, eeg_channel in enumerate(eeg):
    #         artifact_duration = int(fs / artifact_freq)
    #         artifact = np.zeros(len(eeg_channel))
    #         for i in range(0, len(signal), artifact_duration):
    #             artifact[i:i + artifact_duration] = artifact_amplitute
    #         eeg[channel_index] += artifact

    if add_level:
        for channel_index, eeg_channel in enumerate(eeg):
            eeg[channel_index] += channel_index * 5

    return np.array(eeg)


"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

def main(argv):
    eeg = generate_dummy_eeg_signal(89, 2048, 3600, add_level=True)

    srate = 2048
    name = 'BioSemi'
    print('Stream name is ' + name)
    type = 'EEG'
    n_channels = 89
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

    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo(name, type, n_channels, srate, 'float32', 'someuuid1234')

    # next make an outlet
    outlet = StreamOutlet(info)

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0

    sample_index = 0
    while True:
        elapsed_time = local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        for sample_ix in range(required_samples):
            # make a new random n_channels sample; this is converted into a
            # pylsl.vectorf (the data type that is expected by push_sample)
            # mysample = [rand()*10 for _ in range(n_channels)]

            mysample = eeg[:, sample_index]
            # now send it
            outlet.push_sample(mysample)
            sample_index += 1
        sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)


if __name__ == '__main__':
    main(sys.argv[1:])
