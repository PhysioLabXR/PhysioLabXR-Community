import json
import pickle
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eidl.utils.model_utils import get_subimage_model
from numpy import cov
from scipy.special import kl_div, rel_entr
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.decomposition import LatentDirichletAllocation
import textstat

from physiolabxr.scripting.illumiRead.AOIAugmentationScript import AOIAugmentationConfig
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationConfig import study_1_modes, study_2_modes
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationGazeUtils import GazeData, \
    GazeFilterFixationDetectionIVT, \
    GazeType, \
    gaze_point_on_image_valid, tobii_gaze_on_display_area_pixel_coordinate
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import *
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.analysis.utils import get_event_data, \
    get_all_event_conditions_data
from physiolabxr.utils.RNStream import RNStream
from physiolabxr.utils.buffers import DataBuffer


nltk.download('stopwords')
nltk.download('wordnet')

# function definitions ################################################################################################

event_channels = [
    "Block Marker",
    "State Marker",
    "Report Label Marker",
    "AOI Augmentation Start End Marker",
    "Toggle Visual Cue Visibility Marker",
    "Update Visual Cue Marker",
    "Visual Cue History Selected Marker"
  ]

def compute_divergence(user_attention_map, model_attention_map, epsilon = 1e-9):
    user_attention_map = user_attention_map + epsilon
    model_attention_map = model_attention_map + epsilon
    if not np.isclose(np.sum(user_attention_map), 1):
        user_attention_map = user_attention_map / np.sum(user_attention_map)
    if not np.isclose(np.sum(model_attention_map), 1):
        model_attention_map = model_attention_map / np.sum(model_attention_map)
    return rel_entr(model_attention_map, user_attention_map, out=None).sum() / user_attention_map.size

def min_max_normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def process_trial(trial_data: DataBuffer, trial_condition: AOIAugmentationConfig.ExperimentState,
                  experiment_block_images: list, subimage_handler):

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
    original_image_shape = current_image_info.original_image.shape[:2]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gaze_attention_matrix = GazeAttentionMatrix(device=device)
    max_image_size = (3000, 6000)
    gaze_attention_matrix.set_maximum_image_shape(np.array(max_image_size))
    gaze_attention_matrix.setup_gaze_attention_matrix(np.array(original_image_shape))

    ivt_filter = GazeFilterFixationDetectionIVT(angular_speed_threshold_degree=100)

    # construct the fixation sequence
    fixation_sequence = np.zeros_like(gaze_data_ts_stream)  # 0 or 1 depending on if a fixation is made on the image
    fixation_on_image_coordinates = -np.ones((len(gaze_data_ts_stream), 2), dtype=int)  # the fixation coordinates on the image
    for gaze_i, (gaze_data_t, ts) in enumerate(zip(gaze_data_stream.T, gaze_data_ts_stream)):
        # construct gaze data
        gaze_data = GazeData()
        gaze_data.construct_gaze_data_tobii_pro_fusion(gaze_data_t)
        gaze_data_ts_interval = gaze_data.timestamp-ivt_filter.last_gaze_data.timestamp

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
                gaze_on_image_attention_map = gaze_attention_matrix.get_gaze_on_image_attention_map(
                    gaze_point_on_raw_image_coordinate,
                    current_image_info.original_image.shape[:2])  # the gaze attention map on the original image

                gaze_attention_matrix.gaze_attention_pixel_map_clutter_removal(gaze_on_image_attention_map, attention_clutter_ratio=None)  # perform the static clutter removal

                fixation_on_image_duration += gaze_data_ts_interval
                fixation_on_image_coordinates[gaze_i] = gaze_point_on_raw_image_coordinate
                fixation_sequence[gaze_i] = 1

    # identify all the fixations
    fix_onsets = np.where(np.diff(fixation_sequence, prepend=0, append=0) == 1)[0]
    fix_offsets = np.where(np.diff(fixation_sequence, prepend=0, append=0) == -1)[0]

    # fixation_duration_image = np.zeros(current_image_info.original_image.shape[:2])
    # for fix_onset, fix_offset in zip(fix_onsets, fix_offsets):
    #     _fix_coord = fixation_on_image_coordinates[fix_onset]   # fixation coordinate on the r
    #     fixation_duration_image[_fix_coord[0], _fix_coord[1]] += 1  # first is height second is width

    user_attention_mat = gaze_attention_matrix.gaze_attention_pixel_map_buffer.detach().cpu().numpy()

    # get the image attention from the model
    model_name = "vit" if trial_condition == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState or trial_condition == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationInstructionState else "resnet"
    model_attn_mat = subimage_handler.compute_perceptual_attention(
        image_name, is_plot_results=False,  discard_ratio=0.0, model_name=model_name, normalize_by_subimage = False
    )['original_image_attention']

    user_attention_divergence = compute_divergence(user_attention_map=user_attention_mat, model_attention_map=model_attn_mat)

    trial_info['divergence'] = user_attention_divergence
    trial_info['image name'] = image_name

    # get the number of interaction
    trial_info['cue update count'] = np.sum(trial_data['AOIAugmentationEventMarkerLSL'][0][event_channels.index("Update Visual Cue Marker")])
    trial_info['cue history count'] = len([x for x in trial_data['AOIAugmentationEventMarkerLSL'][0][event_channels.index("Visual Cue History Selected Marker")] if x > 0])

    return trial_info


def process_session(data_root, participant_id, sub_image_handler_path):
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

    rn_stream_file_path = None
    for file in os.listdir(participant_folder):
        if file.endswith('.p') and not file.startswith("._"):
            rn_stream_file_path = os.path.join(participant_folder, file)
            break
    assert rn_stream_file_path is not None, "RN stream file does not exist"

    survey_file_path = None
    for file in os.listdir(participant_folder):
        if file.endswith('.csv') and not file.startswith("._"):
            survey_file_path = os.path.join(participant_folder, file)
            break

    assert survey_file_path is not None, "Survey file does not exist"

    print("Start Processing Participant: {}".format(participant_id))

    #  get survey data
    survey_df = pd.read_csv(survey_file_path)
    survey_df["InteractionMode"].tolist()
    # check which study is this
    if set(survey_df["InteractionMode"]) == study_1_modes:
        trial_conditions = [AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState,
                            AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState,  # VIT
                            AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState  # RESNET
                            ]
    elif set(survey_df["InteractionMode"]) == study_2_modes:
        trial_conditions = [AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState,  # VIT
                            AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState,
                            ]
    else:
        raise ValueError("Invalid Study Mode")

    # get the first non practice row
    first_trial = survey_df[survey_df["CurrentBlock"] != "PracticeBlock"]["InteractionMode"].iloc[0]
    is_first_block_no_guidance = first_trial == 'NoAOIAugmentationState'

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

    condition_data = get_all_event_conditions_data(data_buffer=test_block_data,
                                                   # event_enum=AOIAugmentationConfig.ExperimentState,
                                                   event_enum=trial_conditions,
                                                   channel_index=AOIAugmentationConfig.EventMarkerLSLStreamInfo.ExperimentStateChannelIndex)

    experiment_block_images = AOIAugmentationConfig.TestBlockImages
    trial_info_dict = {}
    for trial_condition in trial_conditions:
        trial_info_dict[trial_condition] = []
        for trial_data in condition_data[trial_condition]:
            trial_info = process_trial(trial_data, trial_condition, experiment_block_images, subimage_handler)
            survey_row = survey_df[survey_df["ImageName"] == trial_info["image name"]]
            trial_info["Message"] = survey_row["Message"].values[0]
            trial_info["GlaucomaGroundTruth"] = survey_row["GlaucomaGroundTruth"].values[0]
            trial_info["GlaucomaDecision"] = survey_row["GlaucomaDecision"].values[0]
            trial_info["DecisionConfidenceLevel"] = survey_row["DecisionConfidenceLevel"].values[0]
            trial_info["is first block no guidance"] = is_first_block_no_guidance
            trial_info_dict[trial_condition].append(trial_info)
    pickle.dump(subimage_handler, open(sub_image_handler_path, 'wb'))  # again save the subimage handler, this time with attention cache
    return trial_info_dict, trial_conditions

def process_label(labels):
    return [label.replace("_Suspects", "").replace("_2", "") for label in labels]

def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    # Tokenization
    tokens = text.split()
    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# start of the main block ######################################################
if __name__ == "__main__":  # yes no is has experience ?
    # study_participants = {"04": "Yes",  # # first block no guidance  # STUDY 1
    #                         "08": "No",  # first block no guidance
    #                         "09": "No",  # first block no guidance
    #                         "10": "No",  # first block resnet
    #                         "12": "No",  # first block no guidance
    #                         "13": "Yes",  # first block vit
    #                         "14-1": "Yes",  # first block no guidance
    #                         "15-1": "Yes"}  # first block resnet

    study_participants = {  # STUDY 2
                            "06": "No",
                            "07": "No",
                            "11": "No",
                            "14-2": "Yes",
                            "15-2": "Yes",
    }
    # data_root = "/Users/apocalyvec/PycharmProjects/Temp/AOIAugmentation/"
    m_data_root = r"C:\Dev\PycharmProjects\Temp\AOIAugmentation"
    m_result_root = r"C:\Dev\PycharmProjects\Temp\AOIAugmentation\UserStudyResults 2"
    m_sub_image_handler_path = os.path.join(m_data_root, 'sub_image_handler.p')
    m_data_root = os.path.join(m_data_root, 'Participants')

    reload_session_data = False

    if reload_session_data or not os.path.exists(os.path.join(m_result_root, "sessions.p")):
        sessions = {}
        for m_participant_id in study_participants.keys():
            assert m_participant_id not in sessions, "Participant already processed"
            sessions[m_participant_id], trial_conditions = process_session(m_data_root, m_participant_id, m_sub_image_handler_path)
        pickle.dump(sessions, open(os.path.join(m_result_root, "sessions.p"), 'wb'))
    else:
        sessions = pickle.load(open(os.path.join(m_result_root, "sessions.p"), 'rb'))

    # start of trial analysis ###############################################################################################
    # collect the coherence for each condition
    # create a dataframe for the analysis result, columns are condition, coherence
    trial_results = pd.DataFrame(columns=["participant_id", "Has Experience", "condition", "divergence"])
    for participant_id, session in sessions.items():  # iterate over each participant/session
        for condition, trials in session.items():  # iterate over each condition
            for trial in trials:  # iterate over each trial
                if trial["is first block no guidance"] and condition == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState:
                    trial["condition_additional"] = ["First block no guidance"]
                else:
                    trial["condition_additional"] = [condition.get_name()]
                condition_additional = [condition.get_name()]
                new_row = pd.DataFrame({**{"participant_id": [participant_id],
                                            "Has Experience": study_participants[participant_id],
                                            "condition": [condition.get_name()]},
                                        **trial})
                trial_results = pd.concat([trial_results, new_row], ignore_index=True)
    trial_results.to_csv(os.path.join(m_result_root, "trial_results.csv"))

    trial_results['Text Complexity'] = min_max_normalize(trial_results['Message'].apply(lambda x: -textstat.flesch_reading_ease(x)))

    # analysis of trial data ################################################################################################
    sns.violinplot(data=trial_results, x="condition_additional", y="DecisionConfidenceLevel", hue="Has Experience", split=True, inner="quart")
    plt.xlabel("condition")
    plt.ylim(0)
    plt.show()

    # text complexity
    # sns.violinplot(data=trial_results, x="condition", y="Text Complexity", hue="Has Experience", split=True, inner="quart")
    sns.violinplot(data=trial_results, x="condition", y="Text Complexity", hue="Has Experience")
    # sns.boxplot(data=trial_results, x="condition", y="Text Complexity")
    plt.ylabel("Diagnose insight")
    plt.show()

    # compute p-value between no guidance and vit guidance
    no_guidance = trial_results[trial_results["condition"] == AOIAugmentationConfig.ExperimentState.NoAOIAugmentationState.get_name()]["divergence"]
    reset_guidance = trial_results[trial_results["condition"] == AOIAugmentationConfig.ExperimentState.ResnetAOIAugmentationState.get_name()]["divergence"]
    vit_guidance = trial_results[trial_results["condition"] == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.get_name()]["divergence"]

    u_stat, p_val = stats.mannwhitneyu(trial_results[np.logical_and(trial_results["condition"] == AOIAugmentationConfig.ExperimentState.InteractiveAOIAugmentationState.get_name(),
                                                                    trial_results["Has Experience"] == "No")]["Text Complexity"],
                                       trial_results[np.logical_and(trial_results["condition"] == AOIAugmentationConfig.ExperimentState.StaticAOIAugmentationState.get_name(),
                                                                    trial_results["Has Experience"] == "No")]["Text Complexity"],
                                       alternative='two-sided')

    # session data analysis ################################################################################################
    session_results = pd.DataFrame(columns=["participant_id", "Has Experience", "condition", "divergence", "accuracy", "sensitivity", "specificity", ])
    for participant_id, session in sessions.items():  # iterate over each participant/session
        for condition, trials in session.items():
            avg_divergence = np.mean([trial["divergence"] for trial in trials])
            avg_confidence = np.mean([trial["DecisionConfidenceLevel"] for trial in trials])
            avg_text_complexity = np.mean(trial_results[np.logical_and(trial_results["participant_id"] == participant_id, trial_results["condition"] == condition.get_name())]["Text Complexity"])
            user_decisions = [trial["GlaucomaDecision"] for trial in trials]
            ground_truth = process_label([trial["GlaucomaGroundTruth"] for trial in trials])
            cls_report = classification_report(ground_truth, user_decisions, output_dict=True)
            sensitivity = cls_report["G"]["recall"]
            specificity = cls_report["S"]["recall"]

            new_row = pd.DataFrame({"participant_id": [participant_id],
                                    "Has Experience": study_participants[participant_id],
                                    "condition": [condition.get_name()],
                                    "divergence": [avg_divergence],
                                    "confidence": [avg_confidence],
                                    "accuracy": [cls_report["accuracy"]],
                                    "sensitivity": [sensitivity],
                                    "specificity": [specificity],
                                    "text complexity": [avg_text_complexity]
                                    })
            session_results = pd.concat([session_results, new_row], ignore_index=True)
    session_results.to_csv(os.path.join(m_result_root, "session_results.csv"))

    plt.plot(session_results[session_results["Has Experience"] == "No"]["divergence"], session_results[session_results["Has Experience"] == "No"]["confidence"], 'o')
    plt.show()

    corr, _  = pearsonr(session_results[session_results["Has Experience"] == "No"]["divergence"],
                        session_results[session_results["Has Experience"] == "No"]["confidence"])

    # Latent Dirichlet Allocation
    # trial_results['Processed_Message'] = trial_results['Message'].apply(preprocess_text)
    # vectorizer = CountVectorizer()
    # dtm = vectorizer.fit_transform(trial_results['Processed_Message'])
    #
    # n_topics = 2
    # lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    # lda.fit(dtm)
    #
    # doc_topic_dist = lda.transform(dtm)
    # df_topics = pd.DataFrame(doc_topic_dist, columns=['Topic 1', 'Topic 2'])
    # df_topics["GlaucomaDecision"] = trial_results["GlaucomaDecision"]

