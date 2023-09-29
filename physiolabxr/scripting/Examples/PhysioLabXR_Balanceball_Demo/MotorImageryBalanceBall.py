from enum import Enum

import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.physio.epochs import get_event_locked_data
from physiolabxr.utils.buffers import DataBuffer

event_marker_stream_name = 'EventMarkers'
class GameStates(Enum):
    idle = 'idle'
    train = 'train'
    fit = 'fit'
    eval = 'eval'
    
class Events(Enum):
    train_start = 1
    left_trial = 2
    right_trial = 3
    eval_start = 4

class MotorImageryBalanceBall(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.cur_state = 'idle'

    # Start will be called once when the run button is hit.
    def init(self):
        # self.train_data_buffer = DataBuffer()
        # self.eval_data_buffer = DataBuffer()
        self.transition_markers = [Events.train_start.value, -Events.train_start.value, Events.eval_start.value, -Events.eval_start.value]
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):

        if event_marker_stream_name not in self.inputs.keys(): # or  #EVENT_MARKER_CHANNEL_NAME not in self.inputs.keys():
            print('Event marker stream not found')
            return

        self.process_event_markers()
        if self.cur_state == GameStates.train:
            pass
            # keep collecting data
            # print("In training")
        elif self.cur_state == GameStates.eval:
            # print("In evaluation")
            pass

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def process_event_markers(self):
        if event_marker_stream_name in self.inputs.keys() and np.intersect1d(self.inputs[event_marker_stream_name][0], self.transition_markers):
            last_processed_marker_index = None
            for i, event_marker in enumerate(self.inputs[event_marker_stream_name][0].T):
                game_event_marker = event_marker[0]
                print(f'Event marker is {event_marker} at index {i}')

                # state transition logic
                if game_event_marker == Events.train_start.value:
                    self.cur_state = GameStates.train
                    print('Entering training block')
                    last_processed_marker_index = i

                elif game_event_marker == -Events.train_start.value:  # exiting train state
                    # collect the trials and train the decoding model
                    self.collect_trials_and_train()
                    self.cur_state = GameStates.idle
                    print('Exiting training block')
                    last_processed_marker_index = i

                elif event_marker == Events.eval_start.value:
                    self.cur_state = GameStates.eval
                    print('Entering evaluation block')
                    last_processed_marker_index = i


                elif event_marker == -Events.eval_start.value:
                    self.cur_state = GameStates.idle
                    print('Exiting evaluation block')
                    last_processed_marker_index = i

            # # collect event marker data
            # if self.cur_state == GameStates.train:
            #     event_type = game_state_event_marker
            #     timestamp = self.inputs[event_marker_stream_name][1][i]
            #
            #     # self.train_data_buffer.
            #     pass
            #
            # elif self.cur_state == GameStates.eval:
            #     pass

        # self.inputs.clear_stream_buffer_data(event_marker_stream_name)
            if last_processed_marker_index is not None:
                self.inputs.clear_stream_up_to_index(event_marker_stream_name, last_processed_marker_index+1)

    def collect_trials_and_train(self):
        event_locked_data, last_event_time = get_event_locked_data(event_marker=self.inputs[event_marker_stream_name],
                                                                   data=self.inputs["OpenBCICyton8Channels"],
                                                                   events_of_interest=[Events.left_trial.value, Events.right_trial.value],
                                                                   tmin=0, tmax=5, srate=128, return_last_event_time=True, verbose=1)

