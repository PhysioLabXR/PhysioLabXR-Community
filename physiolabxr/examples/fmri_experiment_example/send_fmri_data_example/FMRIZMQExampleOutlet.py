"""Example program to demonstrate how to send a multi-channel time series to
LSL."""
import time
from collections import deque

import numpy as np
import zmq
from pylsl import local_clock
import pickle
from physiolabxr.examples.fmri_experiment_example.mri_utils import load_nii_gz_file
from scipy.ndimage import zoom

def main():

    # read image to numpy
    # fmri_data = nibabel.load('fmri.nii.gz')
    # image_data = (fmri_data - np.min(fmri_data)) / (np.max(image_data) - np.min(image_data))

    #######_, fmri_data = load_nii_gz_file('resampled_fmri.nii.gz',  normalized=False)
    with open('small_fmri.pickle', 'rb') as handle:
        fmri_data = pickle.load(handle)

    # image = fmri_data[:,:,:, 100]
    # # image = cv2.imread('Image.png')
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # # # image_array = np.array(image)
    # # width = 2555
    # # height = 1246
    # # image = cv2.resize(image, dsize=(int(width*0.5), int(height*0.5)), interpolation=cv2.INTER_CUBIC)
    # # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # # (623, 1277, 3) 2386713
    # image_array = image.flatten()

    topic = "fMRI"
    srate = 2
    port = "5559"

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % port)

    # next make an outlet
    print("now sending data...")
    send_times = deque(maxlen=srate * 10)
    start_time = time.time()
    sent_samples = 0

    image = fmri_data[:, :, :, 10]
    del fmri_data
    # do the image down
    new_matrix = image[:, :, 17:]

    # Pad 5 zero layers to the top
    padding = np.zeros((256, 256, 17))
    image = np.concatenate((new_matrix, padding), axis=-1)
    del new_matrix
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    # zoom image to
    scale_factor = (0.9375, 0.9375, 1.5)

    # Rescale the matrix using zoom
    image = zoom(image, scale_factor)

    #
    # image_array = image.flatten()

    while True:
        elapsed_time = time.time() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        if required_samples > 0:
            # samples = np.random.rand(required_samples * n_channels).reshape((required_samples, -1))
            # samples = (samples * 255).astype(np.uint8)
            for sample_ix in range(required_samples):
                # mysample = samples[sample_ix]
                # random_index = np.random.randint(0, 100)


                # image = fmri_data[:, :, :, random_index]
                # # do the image down
                # new_matrix = image[:,:,:-5]
                #
                # # Pad 5 zero layers to the top
                # padding = np.zeros((256, 256, 5))
                # image = np.concatenate((padding, new_matrix), axis=-1)

                image_array = image.flatten()

                mysample = image_array
                socket.send_multipart([bytes(topic, "utf-8"), np.array(local_clock()), mysample])
                send_times.append(time.time())
            sent_samples += required_samples
        # now send it and wait for a bit before trying again.
        time.sleep(0.01)
        if len(send_times) > 0:
            fps = len(send_times) / (np.max(send_times) - np.min(send_times))
            print("Send FPS is {0}".format(fps), end='\r')

if __name__ == '__main__':
    main()
