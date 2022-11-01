import json

import numpy as np

from rena.scripting.RenaScript import RenaScript


condition_name_dict = {1: "RSVP", 2: "Carousel", 3: "Visual Search", 4: "Table Search"}
metablock_name_dict = {5: "Classifier Prep", 6: "Identifier Prep"}

class ItemEvent():
    """
    event types can be
    1. pop in rsvp
    2. rotation in carousel
    3. gaze in any condition
    4. grab in table search

    duration is a two-item tuple consisting the onset and offset time centered around the event onset
    """
    def __init__(self, data, dtn, event_type, duraiton, item_id, item_distance, ):


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
        self.current_metablock = None
        self.current_condition = None
        self.current_block_id = None
        self.event_marker_channels = json.load(open("../Presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]
        self.loop_count = 0

        self.item_events = []
        self.item_events_queue = []  # the items (timestamps and metainfo of an event) in this queue is emptied to self.item events, when its data is available

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        self.loop_count += 1
        if 'Unity.ReNa.EventMarkers' in self.inputs.buffer.keys():
            new_block_id, new_meta_block, new_condition, block_end = self.get_block_update()

            if block_end:
                try:
                    assert self.current_block_id is not None
                except AssertionError:
                    raise Exception("self.current_block_id is None when block_end signal comes, that means a block start is never received")
                print("[{0}] Block with ID {1} ended. ".format(self.loop_count, self.current_block_id))
            if new_meta_block:
                self.current_block_id = new_block_id
                print("[{0}] Entering new block with ID {1}".format(self.loop_count, metablock_name_dict[self.current_block_id]))
            if new_block_id:
                self.current_block_id = new_block_id
                message = "[{0}] Entering new block with ID {1}".format(self.loop_count, self.current_block_id) + \
                ". No metablock is given, assuming in practice rounds." if not self.current_metablock else ""
                print(message)
            if new_condition:
                self.current_block_id = new_block_id
                print("[{0}] Setting current condition to {1}".format(self.loop_count, condition_name_dict[self.current_block_id]))

            # TODO capture all the DTN events and later process them, so we can clear the event markers first
            dtn_marker = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("DTN"), :]
            # deal with other markers based on the current condition and metablock


            self.inputs.clear_stream_buffer('Unity.ReNa.EventMarkers')  # clear the event markers



        # TODO check if there's any request queued

    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')

    def get_block_update(self):
        # there cannot be multiple condition updates in one block update, which runs once a second
        for i in range(len(self.inputs['Unity.ReNa.EventMarkers'][1])):
            block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), i]
            new_block_id = None if block_id_start_end == 0 or block_id_start_end < 0 else block_id_start_end  # block id less than 0 is also when the block ends

            if block_id_start_end < 0:
                try:
                    assert block_id_start_end == -self.current_block_id
                except AssertionError:
                    raise Exception("Did not receive block end signal")
                block_end = True
            else: block_end = False

            block_marker = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockMarker"), i]
            new_condition = block_marker if block_marker <= 4 else None
            new_meta_block = block_marker if block_marker > 4 else None

        return new_block_id, new_meta_block, new_condition, block_end

    def get_item_event(self):
        for i in range(len(self.inputs['Unity.ReNa.EventMarkers'][1])):
            block_id_start_end = self.inputs['Unity.ReNa.EventMarkers'][0][self.event_marker_channels.index("BlockIDStartEnd"), i]


