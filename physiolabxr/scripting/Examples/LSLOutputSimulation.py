import time

import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript

sampling_rate_key = 'SamplingRate'


class LSLOutputSimulation(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.stream_name = None
        self.invalid = False
        self.last_loop_time = 0

    # Start will be called once when the run button is hit.
    def init(self):
        try:
            assert sampling_rate_key in self.params.keys()
            assert type(self.params[sampling_rate_key]) is int
        except AssertionError:
            print(f"Please add {sampling_rate_key} to parameters as an int")

        try:
            assert len(self.output_names) == 1
            self.stream_name = self.output_names[0]
        except AssertionError:
            print("The outputs must have exactly one stream")
        self.last_loop_time = time.time()


    # loop is called <Run Frequency> times per second
    def loop(self):
        if self.stream_name is not None:
            num_samples_to_send = int((time.time() - self.last_loop_time) * self.params[sampling_rate_key])
            self.last_loop_time = time.time()

            samples_to_send = np.random.rand(num_samples_to_send, self.output_num_channels[self.stream_name])
            self.outputs[self.stream_name] = samples_to_send

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
