import enum

from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.utils.buffers import DataBuffer
import numpy as np

# class TrailInfo:
#     def __init__(self, data_buffer: DataBuffer, condition: AOIAugmentationConfig.ExperimentState):
#         self.data_buffer = data_buffer
#         self.condition = condition
#         image_index = self.get_image_index()
#         if self.condition == AOIAugmentationConfig.ExperimentBlock.PracticeBlock:
#             self.image_id = AOIAugmentationConfig.PracticeBlockImages[image_index]
#         elif self.condition == AOIAugmentationConfig.ExperimentBlock.TestBlock:
#             self.image_id = AOIAugmentationConfig.TestBlockImages[image_index]
#
#     def get_image_index(self):
#         image_id_channel = AOIAugmentationConfig.EventMarkerLSLStreamInfo.ImageIndexChannelIndex
#         image_index = self.data_buffer.get_stream(AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName)[0][image_id_channel, 0]
#         return image_index





def get_event_data(data_buffer: DataBuffer, stream_name, channel_index, event_start_marker, event_end_marker):
    data_buffer_list = []
    event_channel = data_buffer.get_stream(stream_name)[0][channel_index, :]

    event_start_indices = np.where(event_channel == event_start_marker)[0]
    event_end_indices = np.where(event_channel == event_end_marker)[0]

    for event_start_index, event_end_index in zip(event_start_indices, event_end_indices):
        event_start_timestamp = data_buffer.get_stream(stream_name)[1][event_start_index]
        event_end_timestamp = data_buffer.get_stream(stream_name)[1][event_end_index]

        data_buffer_list.append(
            data_buffer.get_all_streams_in_time_range(event_start_timestamp, event_end_timestamp)
        )

    return data_buffer_list


def get_all_event_conditions_data(data_buffer: DataBuffer, event_enum: enum, channel_index):
    event_data = {}
    for event in event_enum:
        event_start_value = event.value
        event_end_value = -event.value

        event_data[event] = get_event_data(
            data_buffer,
            stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
            channel_index=channel_index,
            event_start_marker=event_start_value,
            event_end_marker=event_end_value
        )
    return event_data




