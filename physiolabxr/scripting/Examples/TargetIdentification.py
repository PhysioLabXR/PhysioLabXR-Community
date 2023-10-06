from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from physiolabxr.scripting.RenaScript import RenaScript

item_marker_stream_name = 'Unity.ReNa.ItemMarkers'
event_marker_stream_name = 'Unity.ReNa.EventMarkers'
gaze_pos_stream_name = 'GazePosition'
capture_stream_name = 'CamCapture'
event_marker_block_marker_index = 0
event_marker_block_id_index = 1
item_marker_dtn_index = 0
item_marker_gaze_intersected_index = 5
n_chan_per_item = 10
subimage_size = 128, 128
capture_size = 720, 480

class Events(Enum):
    vs = 3
    ts = 4
    ipe = 6  # identification prep and evaluation

class States(Enum):
    idle = 0  # script isn't doing anything
    vs = 1  # inside of a visual search block

class DTN(Enum):
    distractor = 1
    target = 2
    novelty = 3

class TargetIdentification(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.images = []
        self.cur_state = States.idle
        self.next_state = States.idle
        self.cur_block_id = None

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # print('Loop function is called')
        self.process_eventmarker()

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def process_eventmarker(self):
        """
        This function processes event marker from unity paradigm,
        it implements state transition logics. In-state logic is handled in loop
        """
        if event_marker_stream_name in self.inputs.keys():
            # check if there is a start of an ip meta block
            marker_time = None
            found_event_index = None
            block_markers = self.inputs[event_marker_stream_name][0][event_marker_block_marker_index]
            block_ids = self.inputs[event_marker_stream_name][0][event_marker_block_id_index]

            if len(event_indices := self.get_event_indices(Events.ipe.value)) > 0:
                assert len(event_indices) == 1, "process_eventmarker: there should only be one ipe marker"
                found_event_index = event_indices[0]
                self.images = []  # clear the image buffer
                self.next_state = States.idle
                print(f"found IPD marker, entering {self.next_state}")

            if len(event_indices := self.get_event_indices(Events.vs.value)) > 0:
                assert len(event_indices) == 1, "process_eventmarker: there should only be one vs marker"
                found_event_index = event_indices[0]

                # remember the current block id, we will use it to find when the current block concludes
                self.cur_block_id = block_ids[event_indices[0]]
                self.next_state = States.vs
                print(f"found VS marker, entering {self.next_state}, block id is {self.cur_block_id}")

            if len(event_indices := np.squeeze(np.argwhere(block_ids < 0), axis=1)) > 0:
                assert len(event_indices) == 1, "process_eventmarker: there should only be one vs marker"
                assert (block_end_id := block_ids[event_indices[0]]) == - self.cur_block_id, f"current block id {self.cur_block_id} does not match the block end id {block_end_id}"
                found_event_index = event_indices[0]

                self.next_state = States.idle
                print(f"found block end, entering {self.next_state}, the block just ended is {self.cur_block_id}")

            if found_event_index is not None:
                self.inputs.clear_stream_up_to_index(event_marker_stream_name, found_event_index + 1)

    def get_event_indices(self, event_value):
        block_markers = self.inputs[event_marker_stream_name][0][event_marker_block_marker_index]
        return np.squeeze(np.argwhere(block_markers == event_value), axis=1)

    def get_gaze_intersected_images(self):
        captures = self.inputs[capture_stream_name][0]
        capture_times = self.inputs[capture_stream_name][1]

        gaze_pos = self.inputs[gaze_pos_stream_name][0]
        gaze_pos_times = self.inputs[gaze_pos_stream_name][1]

        item_markers = self.inputs[item_marker_stream_name][0]
        item_markers_times = self.inputs[item_marker_stream_name][1]
        item_dtns = item_markers[item_marker_dtn_index::n_chan_per_item][:, 0]  # TODO verify dtn columns are the same across the times

        item_gaze_intersects = item_markers[item_marker_gaze_intersected_index::n_chan_per_item]
        item_gaze_intersects = np.diff(item_gaze_intersects, axis=1, prepend=0)
        gaze_onsets = np.argwhere(item_gaze_intersects == 1)  # first column is the item index, second column is the time index
        gaze_offsets = np.argwhere(item_gaze_intersects == -1)

        gaze_dtn = item_dtns[gaze_onsets[:, 0]]
        gaze_times = item_markers_times[gaze_onsets[:, 1]]

        # get the gaze positions
        gaze_pos_indices = [np.argmin(np.abs(gaze_pos_times - t)) for t in gaze_times]
        gaze_pos = gaze_pos[:, gaze_pos_indices].T
        # make sure the gaze positions are within the image
        gaze_pos[0] = np.clip(gaze_pos[0], subimage_size[0]/2, capture_size[0] - subimage_size[0]/2)
        gaze_pos[1] = np.clip(gaze_pos[1], subimage_size[1]/2, capture_size[1] - subimage_size[1]/2)

        # get the camera image
        capture_indices = [np.argmin(np.abs(capture_times - t)) for t in gaze_times]
        captures = captures[:, capture_indices].reshape(capture_size[0], capture_size[1], 3, -1)  # three for the color channels
        captures = np.transpose(captures, (3, 0, 1, 2))
        # grab the subimages
        for i, (g_pos, cap) in enumerate(zip(gaze_pos, captures)):
            subimage = cap[int(g_pos[0] - subimage_size[0]/2):int(g_pos[0] + subimage_size[0]/2),
                                  int(g_pos[1] - subimage_size[1]/2):int(g_pos[1] + subimage_size[1]/2), :]
            self.images.append(subimage)
            plt.imsave(f'{i}.png', subimage)

