import time
from collections import deque
import cv2
import numpy
import numpy as np
import time
import os
import pickle
import sys
import matplotlib.pyplot as plt
import pylsl
import torch
from eidl.utils.model_utils import get_subimage_model
from pylsl import StreamInfo, StreamOutlet, cf_float32

from physiolabxr.scripting.fs_utils import get_datetime_str
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, \
    GazeFilterFixationDetectionIVT, \
    tobii_gaze_on_display_area_to_image_matrix_index_when_rect_transform_pivot_centralized, GazeType, \
    gaze_point_on_image_valid, tobii_gaze_on_display_area_pixel_coordinate
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.IntegrateAttention import integrate_attention
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationConfig import EventMarkerLSLStreamInfo, \
    GazeDataLSLStreamInfo, AOIAugmentationScriptParams
import torch
import zmq
from PIL import Image

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


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.current_image_name = None

        # self.vit_attention_matrix = ViTAttentionMatrix()

        self.ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

        self.process_gaze_data_time_buffer = deque(maxlen=1000)

        # self.report_cleaned_image_info_dict = get_report_cleaned_image_info_dict(AOIAugmentationConfig.ReportCleanedImageInfoFilePath, merge_dict=True)

        if os.path.exists(AOIAugmentationConfig.SubImgaeHandlerFilePath):
            with open(AOIAugmentationConfig.SubImgaeHandlerFilePath, 'rb') as f:
                print("Loading subimage handler from pickle")
                self.subimage_handler = pickle.load(f)
        else:
            self.subimage_handler = get_subimage_model()


        self.current_image_info = ImageInfo()

        self.gaze_attention_matrix = GazeAttentionMatrix(device=self.device)

        # prevent null pointer in if statement been compiled before run
        self.gaze_attention_matrix.set_maximum_image_shape(np.array([3000, 6000]))
        # self.gaze_attention_matrix.set_attention_patch_shape(np.array([16, 32]))
        # test = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=False)

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

        self.aoi_augmentation_attention_contour_lsl_outlet = StreamOutlet(
            aoi_augmentation_attention_contour_lsl_outlet_info)  # shape: (1024,)
        # ################################################################################################################
        aoi_augmentation_attention_heatmap_lsl_outlet_info = StreamInfo(
            AOIAugmentationConfig.AOIAugmentationAttentionHeatmapLSLStreamInfo.StreamName,
            AOIAugmentationConfig.AOIAugmentationAttentionHeatmapLSLStreamInfo.StreamType,
            AOIAugmentationConfig.AOIAugmentationAttentionHeatmapLSLStreamInfo.ChannelNum,
            AOIAugmentationConfig.AOIAugmentationAttentionHeatmapLSLStreamInfo.NominalSamplingRate,
            channel_format=cf_float32)

        self.aoi_augmentation_attention_heatmap_lsl_outlet = StreamOutlet(
            aoi_augmentation_attention_heatmap_lsl_outlet_info)

        # ################################################################################################################
        # init heatmap zmq
        self.aoi_augmentation_attention_heatmap_zmq_context = zmq.Context()
        self.aoi_augmentation_attention_heatmap_zmq_socket = self.aoi_augmentation_attention_heatmap_zmq_context.socket(zmq.PUB)
        self.aoi_augmentation_attention_heatmap_zmq_socket.bind("tcp://*:5557")



        # ################################################################################################################
        # self.cur_attention_human = None
        # Start will be called once when the run button is hit.

        self.interactive_aoi_augmentation_log = {
            'fixation_sequence': [],
            'gaze_attention_map': None,
        }

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
        elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState:
            self.resnet_aoi_augmentation_state_callback()

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
            aoi_augmentation_interaction_start_end_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.AOIAugmentationInteractionStartEndMarker]
            toggle_visual_cue_visibility_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.ToggleVisualCueVisibilityMarker]
            update_visual_cue_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.UpdateVisualCueMarker]
            visual_cue_history_selected_marker = event_marker[AOIAugmentationConfig.EventMarkerLSLStreamInfo.VisualCueHistorySelectedMarker]

            # ignore the block_marker <0 and state_marker <0 those means exit the current state
            if block_marker and block_marker > 0:  # evoke block change
                # print(f"entering block {block_marker}")
                self.enter_block(block_marker)

            if state_marker and state_marker > 0:  # evoke state change
                self.enter_state(state_marker)
                print(self.currentExperimentState)
                if state_marker == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState.value or \
                        state_marker == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState.value:

                    # # switch to new interaction state
                    current_image_index = int(image_index_marker)
                    print("current block: {}".format(self.currentBlock))
                    print("current image index: {}".format(current_image_index))
                    if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.PracticeBlock:
                        self.current_image_name = AOIAugmentationConfig.PracticeBlockImages[current_image_index]
                    if self.currentBlock == AOIAugmentationConfig.ExperimentBlock.TestBlock:
                        self.current_image_name = AOIAugmentationConfig.TestBlockImages[current_image_index]

                    print("current report name: {}".format(self.current_image_name))
##########################################################################################################################################################################
                    if self.current_image_name in self.subimage_handler.image_data_dict.keys():
                        current_image_info_dict = self.subimage_handler.image_data_dict[self.current_image_name]
                        current_image_info_dict["image_name"] = self.current_image_name
                        # current_image_attention = self.subimage_handler.compute_perceptual_attention(
                        #     self.current_image_name, is_plot_results=False, discard_ratio=0.0, model_name="vit")
                        #
                        # # merge two dict4
                        # image_info_dict = {**current_image_info_dict, **current_image_attention}
                        self.current_image_info = ImageInfo(**current_image_info_dict)

                        image_on_screen_shape = get_image_on_screen_shape(
                            original_image_width=self.current_image_info.original_image.shape[1],
                            original_image_height=self.current_image_info.original_image.shape[0],
                            image_width=AOIAugmentationConfig.image_on_screen_max_width,
                            image_height=AOIAugmentationConfig.image_on_screen_max_height,
                        )

                        self.current_image_info.image_on_screen_shape = image_on_screen_shape
                        # set image gaze attention matrix buffer


##########################################################################################################################################################################


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
                    elif self.currentExperimentState == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState:
                        self.resnet_aoi_augmentation_state_init_callback()

                    #################################################################################################################

                    self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)  # clear gaze data

            if update_visual_cue_marker==1: # 1 is request received, -1 is request already been updated on the Unity side
                # current_gaze_attention = self.gaze_attention_matrix.get_gaze_attention_grid_map(flatten=False)
                # attention_matrix = self.current_image_info.raw_attention_matrix
                self.update_cue_now = True
                # print("Update Attention Contours")

    def no_aoi_augmentation_state_init_callback(self):

        original_image_rgba = self.current_image_info.get_unreparied_rgba()
        aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
                                                                    image_name=self.current_image_name,
                                                                    image_label=self.current_image_info.label,
                                                                    images_rgba=[original_image_rgba])
        self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)

        pass

    def static_aoi_augmentation_state_init_callback(self):


##########################################################################################################################################################################

        current_image_attention = self.subimage_handler.compute_perceptual_attention(
            self.current_image_name,
            is_plot_results=self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateSubImagePlotWhenUpdate],
            discard_ratio=0.0,
            model_name="vit",
            normalize_by_subimage=
            self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeSubImage]
        )

        self.current_image_info.update_perceptual_image_info(**current_image_attention)

        original_image_rgba = self.current_image_info.get_unreparied_rgba()

        original_image_attention = self.current_image_info.original_image_attention
        original_image_attention_rgba = gray_image_to_rgba(original_image_attention, normalize=True, alpha_threshold=0.9, uint8=True)

        aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
                                                                    image_name=self.current_image_name,
                                                                    image_label=self.current_image_info.label,
                                                                    images_rgba=[original_image_rgba, original_image_attention_rgba])
        self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)
##########################################################################################################################################################################
        print("Visualization Sent")

        # print("Done generating sub images")


        pass

    def interactive_aoi_augmentation_state_init_callback(self):
        self.gaze_attention_matrix.gaze_attention_pixel_map_buffer = torch.tensor(np.zeros(shape=self.current_image_info.original_image.shape[:2]),
                                                                                  device=self.device)
        ##########################################################################################################################################################################

        current_image_attention = self.subimage_handler.compute_perceptual_attention(
            self.current_image_name,
            is_plot_results=self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateSubImagePlotWhenUpdate],
            discard_ratio=0.0,
            model_name="vit",
            normalize_by_subimage=
            self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeSubImage]
        )

        self.current_image_info.update_perceptual_image_info(**current_image_attention)

        original_image_rgba = self.current_image_info.get_unreparied_rgba()

        original_image_attention = self.current_image_info.original_image_attention
        original_image_attention_rgba = gray_image_to_rgba(original_image_attention, normalize=True, alpha_threshold=0.9, uint8=True)

        aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
                                                                    image_name=self.current_image_name,
                                                                    image_label=self.current_image_info.label,
                                                                    images_rgba=[original_image_rgba, original_image_attention_rgba])
        self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)
        ##########################################################################################################################################################################

        pass

    def resnet_aoi_augmentation_state_init_callback(self):
        ##########################################################################################################################################################################

        current_image_attention = self.subimage_handler.compute_perceptual_attention(
            self.current_image_name,
            is_plot_results=self.params[
                AOIAugmentationScriptParams.AOIAugmentationInteractiveStateSubImagePlotWhenUpdate],
            discard_ratio=0.0,
            model_name="resnet",
            normalize_by_subimage=
            self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeSubImage]
        )

        self.current_image_info.update_perceptual_image_info(**current_image_attention)

        original_image_rgba = self.current_image_info.get_unreparied_rgba()

        original_image_attention = self.current_image_info.original_image_attention
        original_image_attention_rgba = gray_image_to_rgba(original_image_attention, normalize=True,
                                                           alpha_threshold=0.9, uint8=True)

        aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(
            topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
            image_name=self.current_image_name,
            image_label=self.current_image_info.label,
            images_rgba=[original_image_rgba, original_image_attention_rgba])
        self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)
        ##########################################################################################################################################################################
        print("Visualization Sent")

        # print("Done generating sub images")

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
        elif state_marker == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationInstructionState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationInstructionState
        elif state_marker == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState.value:
            self.currentExperimentState = AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState
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

            time_start = time.time()

            gaze_data = GazeData()
            gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)

            gaze_data = self.ivt_filter.process_sample(gaze_data)

            # print(gaze_data.gaze_type)

            if gaze_data.combined_eye_gaze_data.gaze_point_valid and gaze_data.gaze_type == GazeType.FIXATION:


                gaze_point_on_screen_pixel_index = tobii_gaze_on_display_area_pixel_coordinate(

                    screen_width=AOIAugmentationConfig.screen_width,
                    screen_height=AOIAugmentationConfig.screen_height,

                    gaze_on_display_area_x=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[0],
                    gaze_on_display_area_y=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[1]
                )
                #
                #
                # # check if on the image
                gaze_point_is_in_screen_image_boundary = gaze_point_on_image_valid(
                    matrix_shape=self.current_image_info.image_on_screen_shape,
                    coordinate=gaze_point_on_screen_pixel_index)
                #
                if gaze_point_is_in_screen_image_boundary:

                    gaze_point_on_raw_image_coordinate = image_coordinate_transformation(
                        original_image_shape=self.current_image_info.image_on_screen_shape,
                        target_image_shape = self.current_image_info.original_image.shape[:2],
                        coordinate_on_original_image=gaze_point_on_screen_pixel_index
                    )

                    # save log
                    self.interactive_aoi_augmentation_log['fixation_sequence'].append(gaze_point_on_raw_image_coordinate)

                    gaze_on_image_attention_map = self.gaze_attention_matrix.get_gaze_on_image_attention_map(
                        gaze_point_on_raw_image_coordinate, self.current_image_info.original_image.shape) # the gaze attention map on the original image
                    # plt.imshow(gaze_on_image_attention_map.detach().cpu().numpy())
                    self.gaze_attention_matrix.gaze_attention_pixel_map_clutter_removal(gaze_on_image_attention_map, attention_clutter_ratio=0.995)  # perform the static clutter removal

            time_end = time.time()

        self.inputs.clear_stream_buffer_data(GazeDataLSLStreamInfo.StreamName)

        if self.update_cue_now:
            print("update cue now")
            gaze_attention_map = self.gaze_attention_matrix.gaze_attention_pixel_map_buffer.detach().cpu().numpy()
            plt.imshow(gaze_attention_map)
            plt.colorbar()
            plt.show()
            if self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeGazeAttention]:
                gaze_attention_map_processed = gaze_attention_map / np.max(gaze_attention_map)
            else:
                gaze_attention_map_processed = gaze_attention_map



            current_image_attention = self.subimage_handler.compute_perceptual_attention(
                self.current_image_name,
                source_attention= gaze_attention_map_processed,
                is_plot_results=self.params[
                AOIAugmentationScriptParams.AOIAugmentationInteractiveStateSubImagePlotWhenUpdate],
                discard_ratio=0.0,
                model_name="vit",
                normalize_by_subimage=
                self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeSubImage]
            )

            self.current_image_info.update_perceptual_image_info(**current_image_attention)

            original_image_rgba = self.current_image_info.get_unreparied_rgba()

            original_image_attention = self.current_image_info.original_image_attention
            original_image_attention_rgba = gray_image_to_rgba(original_image_attention, normalize=True,
                                                               alpha_threshold=0.9, uint8=True)

            gaze_attention_map_rgba = gray_image_to_rgba(gaze_attention_map_processed, normalize=True,
                                                                alpha_threshold=0.9, uint8=True)

            aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(
                topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
                image_name=self.current_image_name,
                image_label=self.current_image_info.label,
                images_rgba=[original_image_rgba, original_image_attention_rgba, gaze_attention_map_rgba])
            self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)


            self.update_cue_now = False

            ##############################################################
            # save log
            self.interactive_aoi_augmentation_log['gaze_attention_map'] = gaze_attention_map
            self.interactive_aoi_augmentation_log['image_info'] = self.current_image_info

            pickle.dump(self.interactive_aoi_augmentation_log, open(f'{self.current_image_name}_{get_datetime_str()}_source-attention-info.p', 'wb'))

            self.interactive_aoi_augmentation_log = {
                'fixation_sequence': [],
                'gaze_attention_map': None,
            }


            ##############################################################
        pass

    def resnet_aoi_augmentation_state_callback(self):
        pass






    def no_attention_callback(self):
        pass

    def static_attention_callback(self):
        pass

    def interactive_attention_callback(self):
        pass


    # def send_zmq_info(self, model_name="vit"):
    #     current_image_attention = self.subimage_handler.compute_perceptual_attention(
    #         self.current_image_name,
    #         is_plot_results=self.params[
    #             AOIAugmentationScriptParams.AOIAugmentationInteractiveStateSubImagePlotWhenUpdate],
    #         discard_ratio=0.0,
    #         model_name=model_name,
    #         normalize_by_subimage=
    #         self.params[AOIAugmentationScriptParams.AOIAugmentationInteractiveStateNormalizeSubImage]
    #     )
    #
    #     self.current_image_info.update_perceptual_image_info(**current_image_attention)
    #
    #     original_image_rgba = self.current_image_info.get_original_image_rgba()
    #
    #     original_image_attention = self.current_image_info.original_image_attention
    #     original_image_attention_rgba = gray_image_to_rgba(original_image_attention, normalize=True,
    #                                                        alpha_threshold=0.9, uint8=True)
    #
    #     aoi_augmentation_multipart = aoi_augmentation_zmq_multipart(
    #         topic="AOIAugmentationAttentionHeatmapStreamZMQInlet",
    #         image_name=self.current_image_name,
    #         image_label=self.current_image_info.label,
    #         images_rgba=[original_image_rgba, original_image_attention_rgba])
    #
    #     self.aoi_augmentation_attention_heatmap_zmq_socket.send_multipart(aoi_augmentation_multipart)



##########################################################################
