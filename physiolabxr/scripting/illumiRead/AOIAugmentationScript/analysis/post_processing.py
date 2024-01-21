from physiolabxr.utils.RNStream import RNStream
import numpy as np
import os
import cv2
import pickle
import random
import pandas as pd
from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
import torch

from physiolabxr.utils.buffers import DataBuffer
from eidl.utils.model_utils import get_subimage_model


participant_id = "01"

sub_image_handler_path = '../data/subimage_handler.pkl'




# load the rn stream data
participant_folder = os.path.join('data', participant_id)

assert os.path.exists(participant_folder), "Participant folder does not exist"



# find the rn stream file end with .dat

rn_stream_file_path = None
for file in os.listdir(participant_folder):
    if file.endswith('.dats'):
        rn_stream_file_path = os.path.join(participant_folder, file)
        break
assert rn_stream_file_path is not None, "RN stream file does not exist"

survey_file_path = None
for file in os.listdir(participant_folder):
    if file.endswith('.csv'):
        survey_file_path = os.path.join(participant_folder, file)
        break

assert survey_file_path is not None, "Survey file does not exist"


print("Start Processing Participant: {}".format(participant_id))


#  get survey data
survey_datafram = pd.read_csv(survey_file_path)

print("Finish Loading Survey Data")


# load the rn stream data
rn_stream = RNStream(rn_stream_file_path)
rn_stream_data = rn_stream.stream_in(jitter_removal=False)

recording_data_buffer = DataBuffer()
recording_data_buffer.buffer = rn_stream_data


# def get_block_start_end_timestamp(data_buffer: DataBuffer, block_enum: AOIAugmentationConfig.ExperimentBlock):
#     event_marker_stream = data_buffer.get_stream(AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName)
#
#     block_events = event_marker_stream[0][AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex, :]
#
#     block_start_index = np.where(block_events == block_enum.value)[0][0]
#
#     block_end_index = np.where(block_events == -block_enum.value)[0][0]
#
#     block_start_timestamp = event_marker_stream[1][block_start_index]
#     block_end_timestamp = event_marker_stream[1][block_end_index]
#
#     return block_start_timestamp, block_end_timestamp
#
# # 610677.294485
# # practice_block_end_timestamp
# # 610862.7931719
#


# practice_block_start_timestamp, practice_block_end_timestamp = get_block_start_end_timestamp(recording_data_buffer, AOIAugmentationConfig.ExperimentBlock.PracticeBlock)
#
# test_block_start_timestamp, test_block_end_timestamp = get_block_start_end_timestamp(recording_data_buffer, AOIAugmentationConfig.ExperimentBlock.TestBlock)
#
# practice_block_data = recording_data_buffer.get_all_streams_in_time_range(practice_block_start_timestamp, practice_block_end_timestamp)
# test_block_data = recording_data_buffer.get_all_streams_in_time_range(test_block_start_timestamp, test_block_end_timestamp)
#
# print("Finish Loading Block Data")
#
# # get data in conditions
#
# # condition data
#
#
# # find all conditions




# event_marker_stream = rn_stream_data[AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName]
#
# block_events = event_marker_stream[0][AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex, :]
#
# practice_block_start_index = np.where(block_events == AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value)[0][0]
#
# practice_block_end_index = np.where(block_events == -AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value)[0][0]
#
# practice_block_start_timestamp = event_marker_stream[1][practice_block_start_index]
# practice_block_end_timestamp = event_marker_stream[1][practice_block_end_index]



# load subimage_handler from pickle

# if os.path.exists(AOIAugmentationConfig.SubImgaeHandlerFilePath):
#     with open(AOIAugmentationConfig.SubImgaeHandlerFilePath, 'rb') as f:
#         subimage_handler = pickle.load(f)
# else:
#     subimage_handler = get_subimage_model()






def event_filter(data_buffer: DataBuffer, stream_name, channel_index, event_start_marker, event_end_marker):
    data_buffer_list = []
    event_channel = data_buffer.get_stream(stream_name)[0][channel_index, :]

    event_start_index = np.where(event_channel == event_start_marker)[0][0]
    event_end_index = np.where(event_channel == event_end_marker)[0][0]

    for event_start_index, event_end_index in zip(event_start_index, event_end_index):
        event_start_timestamp = data_buffer.get_stream(stream_name)[1][event_start_index]
        event_end_timestamp = data_buffer.get_stream(stream_name)[1][event_end_index]

        data_buffer_list.append(
            data_buffer.get_all_streams_in_time_range(event_start_timestamp, event_end_timestamp)
        )

    return data_buffer_list



practice_block_data = event_filter(
    recording_data_buffer,
    stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
    channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex,
    event_start_marker=AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value,
    event_end_marker=-AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value
)

test_block_data = event_filter(
    recording_data_buffer,
    stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
    channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex,
    event_start_marker=AOIAugmentationConfig.ExperimentBlock.TestBlock.value,
    event_end_marker=-AOIAugmentationConfig.ExperimentBlock.TestBlock.value
)

































