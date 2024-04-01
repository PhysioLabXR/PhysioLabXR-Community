import pickle

import pandas as pd
from eidl.utils.model_utils import get_subimage_model

from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, \
    GazeFilterFixationDetectionIVT, \
    GazeType, \
    gaze_point_on_image_valid, tobii_gaze_on_display_area_pixel_coordinate
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.analysis.utils import get_event_data, \
    get_all_event_conditions_data
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.buffers import DataBuffer

participant_id = "01"

# sub_image_handler_path = '../data/subimage_handler.pkl'  # this is where the sub image handler is saved
sub_image_handler_path = '/Users/apocalyvec/PycharmProjects/Temp/AOIAugmentation/subimage_handler.p'
data_root = '/Users/apocalyvec/PycharmProjects/Temp/AOIAugmentation/Participants'


# create a new sub image handler if it does not exist
if os.path.exists(sub_image_handler_path):
    with open(sub_image_handler_path, 'rb') as f:
        subimage_handler = pickle.load(f)
else:
    subimage_handler = get_subimage_model()
    pickle.dump(subimage_handler, open(sub_image_handler_path, 'wb'))


# load the rn stream data
participant_folder = os.path.join(data_root, participant_id)

assert os.path.exists(participant_folder), "Participant folder does not exist"

# find the rn stream file end with .dat

rn_stream_file_path = None
for file in os.listdir(participant_folder):
    if file.endswith('.p'):
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
rn_stream_data = pickle.load(open(rn_stream_file_path, 'rb'))

recording_data_buffer = DataBuffer()
recording_data_buffer.buffer = rn_stream_data

practice_block_data = get_event_data(
    recording_data_buffer,
    stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
    channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex,
    event_start_marker=AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value,
    event_end_marker=-AOIAugmentationConfig.ExperimentBlock.PracticeBlock.value
)

test_block_data = get_event_data(
    recording_data_buffer,
    stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
    channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.BlockChannelIndex,
    event_start_marker=AOIAugmentationConfig.ExperimentBlock.TestBlock.value,
    event_end_marker=-AOIAugmentationConfig.ExperimentBlock.TestBlock.value
)

# get all conditions

practice_block_data = practice_block_data[0]
test_block_data = test_block_data[0]

condition_data = get_all_event_conditions_data(data_buffer=practice_block_data,
                                               event_enum=AOIAugmentationConfig.ExperimentState,
                                               channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex)

trial_conditions = [AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState,
                    AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState,
                    AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState,
                    AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState
                    ]

# gaze on image time, decision time


trial_info_dict = {}



def process_trial(trial_data: DataBuffer, trial_condition: AOIAugmentationConfig.ExperimentState,
                  experiment_block_images: list):

    trial_info = {}

    image_index = trial_data.get_stream(AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName)[0][
                  AOIAugmentationConfig.EventMarkerLSLStreamInfo.ImageIndexChannelIndex, 0].astype(np.int32)

    fixation_on_image_duration = 0

    image_name = experiment_block_images[image_index]
    # get the gaze data
    interaction_data = get_event_data(trial_data, stream_name=AOIAugmentationConfig.EventMarkerLSLStreamInfo.StreamName,
                                    channel_index = AOIAugmentationConfig.EventMarkerLSLStreamInfo.AOIAugmentationInteractionStartEndMarker,
                                    event_start_marker=1,
                                    event_end_marker=-1)[0]


    gaze_stream = interaction_data.get_stream(AOIAugmentationConfig.GazeDataLSLStreamInfo.StreamName)
    gaze_data_stream = gaze_stream[0]
    gaze_data_ts_stream = gaze_stream[1]


    # get current image info
    current_image_info_dict = subimage_handler.image_data_dict[image_name]
    current_image_info = ImageInfo(**current_image_info_dict)
    image_on_screen_shape = get_image_on_screen_shape(
        original_image_width=current_image_info.original_image.shape[1],
        original_image_height=current_image_info.original_image.shape[0],
        image_width=AOIAugmentationConfig.image_on_screen_max_width,
        image_height=AOIAugmentationConfig.image_on_screen_max_height,
    )

    current_image_info.image_on_screen_shape = image_on_screen_shape

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # gaze_attention_matrix = GazeAttentionMatrix(device=device)
    # max_image_size = (3000, 6000)
    # gaze_attention_matrix.set_maximum_image_shape(np.array(max_image_size))

    ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

    for gaze_data_t, ts in zip(gaze_data_stream.T, gaze_data_ts_stream):
        # construct gaze data
        gaze_data = GazeData()
        gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)
        gaze_data_ts_intervel = gaze_data.timestamp-ivt_filter.last_gaze_data.timestamp

        # filter the gaze data with fixation detection
        gaze_data = ivt_filter.process_sample(gaze_data)

        # check if the gaze data is valid or not
        if gaze_data.combined_eye_gaze_data.gaze_point_valid and gaze_data.gaze_type == GazeType.FIXATION:

            # get the gaze point on screen pixel index
            gaze_point_on_screen_pixel_index = tobii_gaze_on_display_area_pixel_coordinate(

                screen_width=AOIAugmentationConfig.screen_width,
                screen_height=AOIAugmentationConfig.screen_height,

                gaze_on_display_area_x=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[0],
                gaze_on_display_area_y=gaze_data.combined_eye_gaze_data.gaze_point_on_display_area[1]
            )

            # check if the gaze point is in the image boundary and is valid and is a fixation
            gaze_point_is_in_screen_image_boundary = gaze_point_on_image_valid(
                matrix_shape=current_image_info.image_on_screen_shape,
                coordinate=gaze_point_on_screen_pixel_index)

            if gaze_point_is_in_screen_image_boundary:

                gaze_point_on_raw_image_coordinate = image_coordinate_transformation(
                    original_image_shape=current_image_info.image_on_screen_shape,
                    target_image_shape=current_image_info.original_image.shape[:2],
                    coordinate_on_original_image=gaze_point_on_screen_pixel_index
                )
                fixation_on_image_duration += gaze_data_ts_intervel



    return trial_info


#

experiment_block_images = AOIAugmentationConfig.PracticeBlockImages

for trial_condition in trial_conditions:
    trial_info_dict[trial_condition] = []
    for trial_data in condition_data[trial_condition]:
        trial_info_dict[trial_condition].append(process_trial(trial_data, trial_condition, experiment_block_images))
