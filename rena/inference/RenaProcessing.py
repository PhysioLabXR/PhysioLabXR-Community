import json

import numpy as np

from rena.scripting.RenaScript import RenaScript


class RenaProcessing(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function

        When the meta block changes
        cp -> cp : train the model, report accuracy and collect more data from the next cp metablock
        cp -> ip: train and exp

        always remove the marker that has been processed from the input buffer
        """
        super().__init__(*args, **kwargs)
        self.current_meta_block = None
        self.current_condition = None
        self.current_block_id = None
        self.event_marker_channels = json.load(open("../Presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        if 'Unity.ReNa.EventMarkers' in self.inputs.buffer.keys():
            block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), :]
            dtn_marker = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("DTN"), :]

            if np.any(block_id_start_end != 0):
                temp = np.argwhere(block_id_start_end != 0)[0]
                assert len(temp) == 1  # there can only be one block start give
                self.current_block_id = block_id_start_end[temp.item]
                print("Entered block with ID {}".format(self.current_block_id))
                block_id_start_end[temp.item] = 0

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
