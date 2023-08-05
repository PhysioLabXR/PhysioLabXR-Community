from rena.scripting.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, GazeFilterFixationDetectionIVT
from rena.scripting.RenaScript import RenaScript
from rena.scripting.AOIAugmentationScript import AOIAugmentationConfig
from rena.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
from rena.scripting.AOIAugmentationScript.AOIAugmentationConfig import EventMarkerLSLStreamInfo, GazeDataLSLStreamInfo
import torch


class AOIAugmentationScript(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)
        self.currentExperimentState: AOIAugmentationConfig.ExperimentState = \
            AOIAugmentationConfig.ExperimentState.StartState

        self.currentBlock: AOIAugmentationConfig.ExperimentBlock = \
            AOIAugmentationConfig.ExperimentBlock.StartBlock

        self.currentReportLabel: int = -1

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.attentionMatrix = AOIAttentionMatrixTorch(
            attention_matrix=None,
            image_shape=AOIAugmentationConfig.image_shape,
            attention_patch_shape=AOIAugmentationConfig.attention_grid_shape,
            device=self.device
        )

        # gaze data component
        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)
        self.ivt_filter.evoke_data_processor()

    # Start will be called once when the run button is hit.
    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):
        # fixation_detection
        # list of gaze of intersection
        # show pixel on patch x
        # detected_fixation_on_display_area = [1000, 1000]

        # state machine
        # gaze_on_screen_area = [0.12, 0.1]
        # print('Loop function is called')

        # if EventMarkerLSLOutlet.StreamName not in self.inputs.keys() or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
        #     return

        if EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys(): # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
            return

        self.state_shift()

        # if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.:
        #     self.practice_state()
        #
        # # if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.PracticeState:
        # #     self.practice_state()

        if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState:
            self.no_aoi_augmentation_state_callback()
        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState:
            self.static_aoi_augmentation_state_callback()
        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState:
            self.interactive_aoi_augmentation_state_callback()

        # if InterruptExperimentMarker in self.inputs.get_data(P300EventStreamName)[
        #     p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:

        # if InterruptExperimentMarker in self.inputs.get_data(P300EventStreamName)[
        #     p300_speller_event_marker_channel_index['P300SpellerGameStateControlMarker']]:

        # event_marker = self.inputs[EventMarkerLSLOutlet.StreamName].pop()

        # if not in  #self.inputs.keys() or EventMarkerLSLOutlet not in self.inputs.keys():
        #     return

        # practice
        #
        #
        #
        #
        # experiment

        # check if we are in the static interaction mode

        # state machine

    # cleanup is called when the stop button is hit

    def cleanup(self):
        print('Cleanup function is called')

    def state_shift(self):
        event_markers = self.inputs[EventMarkerLSLStreamInfo.StreamName][0]
        self.inputs.clear_stream_buffer(EventMarkerLSLStreamInfo.StreamName)

        # state shift
        for event_marker in event_markers.T:
            block_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex]
            state_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex]
            report_label_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ReportLabelChannelIndex]

            if block_marker and block_marker>0: # evoke block change
                self.enter_block(block_marker)


            if state_marker and state_marker>0: # evoke state change
                self.enter_state(state_marker)
                print(self.currentExperimentState)
                if state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState or \
                        state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState or \
                        state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState:
                    # switch to new interaction state
                    currentReportLabel = report_label_marker
                    print("set report interaction label to {}".format(currentReportLabel))
                    self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName) # clear gaze data

        # AOI state machine
        if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState:
            self.no_aoi_augmentation_state_callback()

        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState:
            self.static_aoi_augmentation_state_callback()

        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState:
            self.interactive_aoi_augmentation_state_callback()

        else:
            # no interaction required
            pass



    def enter_block(self, block_marker):
        if block_marker == AOIAugmentationConfig.ExperimentBlock.InitBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.InitBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.StartBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.StartBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.IntroductionBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.IntroductionBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.PracticeBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.ExperimentBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.ExperimentBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.EndBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.EndBlock
        else:
            print("Invalid block marker")
            # raise ValueError('Invalid block marker')

    def enter_state(self, state_marker):
        if state_marker == AOIAugmentationConfig.ExperimentState.CalibrationState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.CalibrationState
        elif state_marker == AOIAugmentationConfig.ExperimentState.StartState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.StartState
        elif state_marker == AOIAugmentationConfig.ExperimentState.IntroductionInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.IntroductionInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.PracticeInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.PracticeInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.NoAOIAugmentationInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState
        elif state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState
        elif state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState
        elif state_marker == AOIAugmentationConfig.ExperimentState.FeedbackState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.FeedbackState
        elif state_marker == AOIAugmentationConfig.ExperimentState.EndState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.EndState
        else:
            print("Invalid state marker")
            # raise ValueError('Invalid state marker')

    def no_aoi_augmentation_state_callback(self):
        pass

    def static_aoi_augmentation_state_callback(self):
        # process the eye tracking data
        # run real-time fixation detection algorithm
        pass

    def interactive_aoi_augmentation_state_callback(self):
        pass

    def clear_eye_tracking_data(self):

        print("clear eye tracking data")
        pass
    def attention_map_callback(self):
        for gaze_data_t in self.inputs[GazeDataLSLStreamInfo.StreamName][1]:
            # construct gaze data
            gaze_data = GazeData()
            gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)
            gaze_data = self.ivt_filter.process_sample(gaze_data)
            # the gaze data has been classified at this point

            # 1. calculate the fixation location on the image
            # 2.



            # based on if this is a fixation, we define
            #
            # get the fixation on screen position



            # print(gaze_data)
            # construct gaze data

            pass



