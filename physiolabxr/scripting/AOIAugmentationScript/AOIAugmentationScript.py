import time
from collections import deque
import cv2
import numpy as np
import time
import os
import pickle
import sys
import matplotlib.pyplot as plt
import torch

from pylsl import StreamInfo, StreamOutlet, cf_float32


from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, GazeFilterFixationDetectionIVT, \
    tobii_gaze_on_display_area_to_image_matrix_index, GazeType, gaze_point_on_image_valid
from physiolabxr.scripting.AOIAugmentationScript.IntegrateAttention import integrate_attention
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
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

        self.device = torch.device('cpu')


        self.current_image_name = None


        # self.vit_attention_matrix = ViTAttentionMatrix()

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        self.report_cleaned_image_info_dict = get_report_cleaned_image_info_dict(AOIAugmentationConfig.ReportCleanedImageInfoFilePath, merge_dict=True)

        self.gaze_attention_matrix = GazeAttentionMatrix(device=self.device)

        # prevent null pointer in if statement been compiled before run
        self.gaze_attention_matrix.set_image_shape(np.array([512,1024]))
        self.gaze_attention_matrix.set_attention_patch_shape(np.array([16,32]))
        test = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=False)

        self.current_image_info = ImageInfo()
        self.update_cue_now = False

        print("Experiment started")
        # self.vit_attention_matrix.generate_random_attention_matrix(patch_num=1250)

        # self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)
        # self.ivt_filter.evoke_data_processor()


        # ################################################################################################################
        gaze_attention_map_lsl_outlet_info = StreamInfo(
            AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.StreamName,
            AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.StreamType,
            AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.ChannelNum,
            AOIAugmentationConfig.AOIAugmentationGazeAttentionMapLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.gaze_attention_map_lsl_outlet = StreamOutlet(gaze_attention_map_lsl_outlet_info)  # shape: (1024, 1)
        #
        # ################################################################################################################


        # ################################################################################################################
        aoi_augmentation_attention_contour_lsl_outlet_info = StreamInfo(
            AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.StreamName,
            AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.StreamType,
            AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.ChannelNum,
            AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.aoi_augmentation_attention_contour_lsl_outlet = StreamOutlet(aoi_augmentation_attention_contour_lsl_outlet_info)  # shape: (1024,)
        # ################################################################################################################

        # self.cur_attention_human = None
        # Start will be called once when the run button is hit.


    def init(self):
        pass

    # loop is called <Run Frequency> times per second
    def loop(self):



        if (EventMarkerLSLStreamInfo.StreamName not in self.inputs.keys()) or (
                GazeDataLSLStreamInfo.StreamName not in self.inputs.keys()):  # or GazeDataLSLOutlet.StreamName not in self.inputs.keys():
            return
        # print("process event marker call start")
        self.process_event_markers()
        # print("process event marker call complete")

        if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState:
            self.no_aoi_augmentation_state_callback()
        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState:
            self.static_aoi_augmentation_state_callback()
        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState:
            self.interactive_aoi_augmentation_state_callback()

    def cleanup(self):
        print('Cleanup function is called')

    def process_event_markers(self):
        event_markers = self.inputs[EventMarkerLSLStreamInfo.StreamName][0]
        self.inputs.clear_stream_buffer_data(EventMarkerLSLStreamInfo.StreamName)

        # state shift
        for event_marker in event_markers.T:
            # print(f"working on event marker {event_marker}")
            block_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex]
            state_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex]
            image_index_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ImageIndexChannelIndex]
            user_inputs_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.UserInputsChannelIndex]

            # ignore the block_marker <0 and state_marker <0 those means exit the current state
            if block_marker and block_marker > 0:  # evoke block change
                # print(f"entering block {block_marker}")
                self.enter_block(block_marker)

            if state_marker and state_marker > 0:  # evoke state change
                self.enter_state(state_marker)
                print(self.currentExperimentState)
                if state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState.value:

                    # # switch to new interaction state
                    current_image_index = int(image_index_marker)
                    print("current block: {}".format(self.currentBlock))
                    print("current image index: {}".format(current_image_index))
                    if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.PracticeBlock:
                        self.current_image_name = AOIAugmentationConfig.PracticeBlockImages[current_image_index]
                    if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.TestBlock:
                        self.current_image_name = AOIAugmentationConfig.TestBlockImages[current_image_index]

                    print("current report name: {}".format(self.current_image_name))

                    if self.current_image_name in self.report_cleaned_image_info_dict.keys():
                        current_image_info_dict = self.report_cleaned_image_info_dict[self.current_image_name]
                        self.current_image_info = ImageInfo(**current_image_info_dict)

                        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        self.gaze_attention_matrix = GazeAttentionMatrix(device=self.device)
                        self.gaze_attention_matrix.set_image_shape(self.current_image_info.model_image_shape)
                        self.gaze_attention_matrix.set_attention_patch_shape(self.current_image_info.patch_shape)

                        # get image on screen pixel shape
                        image_on_screen_shape = get_image_on_screen_shape(
                            original_image_width=self.current_image_info.original_image.shape[1],
                            original_image_height=self.current_image_info.original_image.shape[0],
                            image_width=AOIAugmentationConfig.image_on_screen_width,
                            image_height=AOIAugmentationConfig.image_on_screen_height,
                        )

                        self.current_image_info.image_on_screen_shape = image_on_screen_shape

                    else:
                        self.current_image_info = None
                        print("image info not found in dict")
                        # stop the experiment

                    if self.currentExperimentState == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState:
                        self.no_aoi_augmentation_state_init_callback()
                    elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState:
                        self.static_aoi_augmentation_state_init_callback()
                    elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState:
                        self.interactive_aoi_augmentation_state_init_callback()

#################################################################################################################

                    self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)  # clear gaze data

            if user_inputs_marker == AOIAugmentationConfig.UserInputTypes.AOIAugmentationInteractionStateUpdateCueKeyPressed.value:
                # current_gaze_attention = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=False)
                # attention_matrix = self.current_image_info.raw_attention_matrix
                self.update_cue_now = True
                print("Update Attention Contours")




    def no_aoi_augmentation_state_init_callback(self):
        pass

    def static_aoi_augmentation_state_init_callback(self):
        # register the image on screen shape
        start = time.time()

        # calculate the contour
        heatmap_processed = cv2.resize(self.current_image_info.rollout_attention_matrix,
                                       dsize=(self.current_image_info.image_on_screen_shape[1], self.current_image_info.image_on_screen_shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        ret, thresh = cv2.threshold(heatmap_processed, 0.5, 1, 0)
        # apply erosion to threshold image
        kernel = np.ones((5, 5), np.uint8)

        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # convert contours coordinate to image on screen coordinate

        # compile the contour information to lvt

        contours_lvt, overflow_flag = contours_to_lvt(contours, hierarchy,
                                                      max_length=AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.ChannelNum)
        # self.aoi_augmentation_attention_contour_lsl_outlet.push_sample(contours_lvt)

        # TODO: send the contour information to Unity
        self.aoi_augmentation_attention_contour_lsl_outlet.push_sample(contours_lvt)
        print("time for contour: {}".format(time.time() - start))

    def interactive_aoi_augmentation_state_init_callback(self):
        # register the image on screen shape
        start = time.time()

        # calculate the contour
        heatmap_processed = cv2.resize(self.current_image_info.rollout_attention_matrix,
                                       dsize=(self.current_image_info.image_on_screen_shape[1], self.current_image_info.image_on_screen_shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
        ret, thresh = cv2.threshold(heatmap_processed, 0.5, 1, 0)
        # apply erosion to threshold image
        kernel = np.ones((5,5), np.uint8)

        thresh = cv2.erode(thresh, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # convert contours coordinate to image on screen coordinate

        # compile the contour information to lvt

        contours_lvt, overflow_flag = contours_to_lvt(contours, hierarchy,
                                                      max_length=AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.ChannelNum)
        # self.aoi_augmentation_attention_contour_lsl_outlet.push_sample(contours_lvt)

        # TODO: send the contour information to Unity
        self.aoi_augmentation_attention_contour_lsl_outlet.push_sample(contours_lvt)
        print("time for contour: {}".format(time.time() - start))

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

        for gaze_data_t in self.inputs[GazeDataLSLStreamInfo.StreamName][0].T:
            gaze_data = GazeData()
            gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)

            gaze_data = self.ivt_filter.process_sample(gaze_data)

            # print(gaze_data.gaze_type)

            if gaze_data.combined_eye_gaze_data.gaze_point_valid and gaze_data.gaze_type == GazeType.FIXATION:

                gaze_point_on_screen_image_index = tobii_gaze_on_display_area_to_image_matrix_index(
                    image_center_x=AOIAugmentationConfig.image_center_x,
                    image_center_y=AOIAugmentationConfig.image_center_y,

                    image_width=self.current_image_info.image_on_screen_shape[1],
                    image_height=self.current_image_info.image_on_screen_shape[0],

                    screen_width=AOIAugmentationConfig.screen_width,
                    screen_height=AOIAugmentationConfig.screen_height,

                    gaze_on_display_area_x=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[0],
                    gaze_on_display_area_y=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[1]
                )


                # check if on the image
                gaze_point_is_in_screen_image_boundary = gaze_point_on_image_valid(
                    matrix_shape=self.current_image_info.image_on_screen_shape,
                    coordinate=gaze_point_on_screen_image_index)

                if gaze_point_is_in_screen_image_boundary:
                    gaze_point_on_model_coordinate = coordinate_transformation(
                        original_image_shape=self.current_image_info.image_on_screen_shape,
                        target_image_shape = self.current_image_info.model_image_shape,
                        coordinate_on_original_image=gaze_point_on_screen_image_index
                    )

                    gaze_on_image_attention_map = self.gaze_attention_matrix.get_gaze_on_image_attention_map(gaze_point_on_model_coordinate)
                    gaze_on_grid_attention_map = self.gaze_attention_matrix.get_patch_attention_map(gaze_on_image_attention_map)

                    self.gaze_attention_matrix.gaze_attention_grid_map_clutter_removal(gaze_on_grid_attention_map, attention_clutter_ratio=0.995)

                    gaze_attention_grid_map = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=True)

                    self.gaze_attention_map_lsl_outlet.push_sample(gaze_attention_grid_map)

                # gaze_point_on_screen_image = tobii_gaze_on_display_area_to_image_matrix_index(
                #     image_center_
                # )

        # self.cur_attention_human = gaze_attention_grid_map

        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)

        if self.update_cue_now:
            current_gaze_attention = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=False)
            attention_matrix = self.current_image_info.raw_attention_matrix
            integrated_attention = integrate_attention(attention_human=current_gaze_attention.reshape(-1), attention_vit=attention_matrix).reshape(32, 32)
            integrated_attention -= integrated_attention.min()
            integrated_attention /= integrated_attention.max() - integrated_attention.min()

            integrated_attention_resized = cv2.resize(integrated_attention,
                                           dsize=(self.current_image_info.image_on_screen_shape[1],
                                                  self.current_image_info.image_on_screen_shape[0]),
                                           interpolation=cv2.INTER_NEAREST)

            ret, thresh = cv2.threshold(integrated_attention_resized, self.params['cue_threshold'], 1, 0)
            # apply erosion to threshold image
            kernel = np.ones((5,5), np.uint8)

            thresh = cv2.erode(thresh, kernel, iterations=1)
            contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # convert contours coordinate to image on screen coordinate

            # compile the contour information to lvt

            contours_lvt, overflow_flag = contours_to_lvt(contours, hierarchy,
                                                          max_length=AOIAugmentationConfig.AOIAugmentationAttentionContourLSLStreamInfo.ChannelNum)

            # TODO: send the contour information to Unity
            self.aoi_augmentation_attention_contour_lsl_outlet.push_sample(contours_lvt)

            self.update_cue_now = False

        pass

        # self.interactive_attention_callback()
        # pass




##########################################################################
    def no_attention_callback(self):
        pass

    def static_attention_callback(self):
        pass


    def interactive_attention_callback(self):
        pass

##########################################################################
