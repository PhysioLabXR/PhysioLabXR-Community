from collections import deque

from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeFilterFixationDetectionIVT
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import EventMarkerLSLStreamInfo, GazeDataLSLStreamInfo
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

        # self.gaze_attention_matrix = GazeAttentionMatrixTorch(
        #     image_shape=AOIAugmentationConfig.image_shape,
        #     attention_patch_shape=AOIAugmentationConfig.attention_patch_shape,
        #     device=self.device
        # )
        # self.gaze_attention_clutter_removal_data_processor = ClutterRemoval(signal_clutter_ratio=0.1)
        # self.gaze_attention_clutter_removal_data_processor.evoke_data_processor()

        self.vit_attention_matrix = ViTAttentionMatrix()
        # self.vit_attention_matrix.generate_random_attention_matrix(patch_num=1250)

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)
        self.ivt_filter.evoke_data_processor()

        # ################################################################################################################
        # static_aoi_augmentation_lsl_outlet_info = StreamInfo(
        #     AOIAugmentationConfig.StaticAOIAugmentationStateLSLStreamInfo.StreamName,
        #     AOIAugmentationConfig.StaticAOIAugmentationStateLSLStreamInfo.StreamType,
        #     AOIAugmentationConfig.StaticAOIAugmentationStateLSLStreamInfo.ChannelNum,
        #     AOIAugmentationConfig.StaticAOIAugmentationStateLSLStreamInfo.NominalSamplingRate,
        #     channel_format=cf_float32)
        # self.static_aoi_augmentation_lsl_outlet = StreamOutlet(static_aoi_augmentation_lsl_outlet_info)
        #
        # ################################################################################################################
        # interactive_aoi_augmentation_lsl_outlet_info = StreamInfo(
        #     AOIAugmentationConfig.InteractiveAOIAugmentationStateLSLStreamInfo.StreamName,
        #     AOIAugmentationConfig.InteractiveAOIAugmentationStateLSLStreamInfo.StreamType,
        #     AOIAugmentationConfig.InteractiveAOIAugmentationStateLSLStreamInfo.ChannelNum,
        #     AOIAugmentationConfig.InteractiveAOIAugmentationStateLSLStreamInfo.NominalSamplingRate,
        #     channel_format=cf_float32)
        # self.interactive_aoi_augmentation_lsl_outlet = StreamOutlet(interactive_aoi_augmentation_lsl_outlet_info)
        # ################################################################################################################
        #
        # ################################################################################################################
        # gaze_attention_map_lsl_outlet_info = StreamInfo(
        #     AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.StreamName,
        #     AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.StreamType,
        #     AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.ChannelNum,
        #     AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.NominalSamplingRate,
        #     channel_format=cf_float32)
        #
        # self.gaze_attention_map_lsl_outlet = StreamOutlet(gaze_attention_map_lsl_outlet_info)  # shape: (1250, 1)
        #
        # ################################################################################################################

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        ################################################################################################################
        self.practice_block_attention_map = {}
        self.test_block_attention_map = {}
    # Start will be called once when the run button is hit.
    def init(self):

        pass
        # self.practice_block_attention_map = get_all_attention_matrices(
        #     image_directory=AOIAugmentationConfig.PracticeBlockImageDirectoryPath,
        #     image_shape=AOIAugmentationConfig.image_shape,
        #     attention_patch_shape=AOIAugmentationConfig.attention_patch_shape
        # )

        # self.test_block_attention_map = get_all_attention_matrices(
        #     image_directory=AOIAugmentationConfig.TestBlockImageDirectoryPath,
        #     image_shape=AOIAugmentationConfig.image_shape,
        #     attention_patch_shape=AOIAugmentationConfig.attention_patch_shape
        # )

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

        if (EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()):  # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
            # print("EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys() or GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()")
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

    def cleanup(self):
        print('Cleanup function is called')

    def state_shift(self):
        event_markers = self.inputs[EventMarkerLSLStreamInfo.StreamName][0]
        self.inputs.clear_stream_buffer_data(EventMarkerLSLStreamInfo.StreamName)

        # state shift
        for event_marker in event_markers.T:
            block_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex]
            state_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex]
            image_index_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ImageIndexChannelIndex]
            interrupt_label_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.UpdateVisualCueMarker]

            # ignore the block_marker <0 and state_marker <0 those means exit the current state
            if block_marker and block_marker > 0:  # evoke block change
                self.enter_block(block_marker)

            if state_marker and state_marker > 0:  # evoke state change
                self.enter_state(state_marker)
                print(self.currentExperimentState)
                if state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState.value:

                    # # switch to new interaction state
                    # current_image_index = int(image_index_marker)
                    # print("current block is {}".format(self.currentBlock))
                    # print("current image index is {}".format(current_image_index))
                    # # set attention matrix
                    # if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.PracticeBlock:
                    #     self.vit_attention_matrix.set_attention_matrix(self.practice_block_attention_map[current_image_index])
                    # if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.TestBlock:
                    #     self.vit_attention_matrix.set_attention_matrix(self.test_block_attention_map[current_image_index])
                    #
                    # ######
                    # self.vit_attention_matrix.calculate_patch_average_attention_vector()

                    self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)  # clear gaze data

    def enter_block(self, block_marker):
        if block_marker == AOIAugmentationConfig.ExperimentBlock.InitBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.InitBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.StartBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.StartBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.IntroductionBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.IntroductionBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.PracticeBlock
        elif block_marker == AOIAugmentationConfig.ExperimentBlock.TestBlock.value:
            self.currentBlock = AOIAugmentationConfig.ExperimentBlock.TestBlock
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
        elif state_marker == AOIAugmentationConfig.ExperimentState.TestInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.TestInstructionState
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
        self.no_attention_callback()
        pass

    def static_aoi_augmentation_state_callback(self):
        self.static_attention_callback()
        pass

    def interactive_aoi_augmentation_state_callback(self):
        self.interactive_attention_callback()
        pass

    def no_attention_callback(self):
        pass

    def static_attention_callback(self):
        pass

    # def static_attention_callback_archive(self):

        # for gaze_data_t in self.inputs[GazeDataLSLStreamInfo.StreamName][0].T:
        #     gaze_data_process_start = time.time()
        #
        #     self.gaze_attention_matrix.reset_image_attention_buffer()
        #     self.gaze_attention_matrix.reset_attention_grid_buffer()
        #
        #     # print("process gaze data")
        #
        #     # construct gaze data
        #     gaze_data = GazeData()
        #     gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)
        #
        #     gaze_data = self.ivt_filter.process_sample(gaze_data)  # ivt filter
        #
        #     if gaze_data.combined_eye_gaze_data.gaze_point_valid and gaze_data.gaze_type == GazeType.FIXATION:
        #         # gaze point is valid and gaze type is fixation
        #         # check if gaze point is in screen image boundary
        #         gaze_point_on_screen_image = tobii_gaze_on_display_area_to_image_matrix_index(
        #             image_center_x=AOIAugmentationConfig.image_center_x,
        #             image_center_y=AOIAugmentationConfig.image_center_y,
        #
        #             image_width=AOIAugmentationConfig.image_on_screen_width,
        #             image_height=AOIAugmentationConfig.image_on_screen_height,
        #
        #             screen_width=AOIAugmentationConfig.screen_width,
        #             screen_height=AOIAugmentationConfig.screen_height,
        #
        #             gaze_on_display_area_x=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[0],
        #             gaze_on_display_area_y=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[1]
        #         )
        #
        #         gaze_point_is_in_screen_image_boundary = gaze_point_on_image_valid(
        #             matrix_shape=AOIAugmentationConfig.image_on_screen_shape,
        #             coordinate=gaze_point_on_screen_image)
        #
        #         if gaze_point_is_in_screen_image_boundary:
        #             gaze_point_on_image = np.floor(
        #                 gaze_point_on_screen_image / AOIAugmentationConfig.image_scaling_factor).astype(int)
        #             self.gaze_attention_matrix.get_image_attention_buffer(
        #                 gaze_point_on_image)  # get attention position G
        #             self.gaze_attention_matrix.convolve_attention_grid_buffer()  # calculate attention grid using convolve
        #
        #     # clutter removal
        #     self.gaze_attention_matrix.gaze_attention_grid_map_clutter_removal(attention_clutter_ratio=0.95)
        #     self.send_static_aoi_augmentation_state_lsl(gaze_attention_threshold=0.1)  # send the attention vector
        #     self.process_gaze_data_time_buffer.append(time.time() - gaze_data_process_start)
        #
        #     # calculate average process time of process gaze data time buffer
        #     # average_time = np.mean(self.process_gaze_data_time_buffer)
        #     # print("average_process_gaze_data_time:", average_time)
        #
        #     # push attention map for visualization
        #     gaze_attention_vector = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=True)
        #     self.gaze_attention_map_lsl_outlet.push_sample(gaze_attention_vector)
        #
        #     threshold_vit_attention_vector = self.vit_attention_matrix.threshold_patch_average_attention(threshold=0.52)
        #     # mask vit attention with gaze attention
        #
        # self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)
        # pass

    def interactive_attention_callback(self):
        # TODO: implement interactive attention callback
        pass





    # def send_static_aoi_augmentation_state_lsl(self, gaze_attention_threshold=0.1):
    #     gaze_attention_vector = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=True)
    #     threshold_vit_attention_vector = self.vit_attention_matrix.threshold_patch_average_attention(threshold=0.52)
    #     # mask vit attention with gaze attention
    #     vit_attention_vector_mask = np.where(gaze_attention_vector > gaze_attention_threshold, 0, 1)
    #     self.static_aoi_augmentation_lsl_outlet.push_sample(vit_attention_vector_mask * threshold_vit_attention_vector)

