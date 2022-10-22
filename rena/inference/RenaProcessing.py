import json

import numpy as np

from rena.scripting.RenaScript import RenaScript


class RenaProcessing(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.current_block = None
        self.event_marker_channels = json.load(open("Presets/LSLPresets/ReNaEventMarker.json", 'r'))["ChannelNames"]

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        #
        self.inputs['Unity.ReNa.EventMarkers'][self.event_marker_channels.index("DTN")]


    # cleanup is called when the stop button is hit
    def cleanup(self):
        print('Cleanup function is called')
