import numpy as np
import pandas as pd
from physiolabxr.utils.data_utils import RNStream
import pickle

varjo_channels = [
    "raw_timestamp",
    "log_time",
    "focus_distance",
    "frame_number",
    "stability",
    "status",
    "Angle2CameraUp",
    "gaze_forward_x",
    "gaze_forward_y",
    "gaze_forward_z",
    "gaze_origin_x",
    "gaze_origin_y",
    "gaze_origin_z",
    "HMD_position_x",
    "HMD_position_y",
    "HMD_position_z",
    "HMD_rotation_x",
    "HMD_rotation_y",
    "HMD_rotation_z",
    "left_forward_x",
    "left_forward_y",
    "left_forward_z",
    "left_origin_x",
    "left_origin_y",
    "left_origin_z",
    "left_pupil_size",
    "left_status",
    "right_forward_x",
    "right_forward_y",
    "right_forward_z",
    "right_origin_x",
    "right_origin_y",
    "right_origin_z",
    "right_pupil_size",
    "right_status"
]


def gaze_vector_to_screen_space(gaze_vector, gaze_origin, screen_width, screen_height,
                                normalized=True):  # origin top left conor


    # # calculate x
    x_offset = (gaze_vector[0] / gaze_vector[2]) * gaze_origin[2]
    y_offset = (gaze_vector[1] / gaze_vector[2]) * gaze_origin[2]

    gaze_x_location = gaze_origin[0]+x_offset
    gaze_y_location = gaze_origin[1]+y_offset

    if normalized:
        gaze_x_location = gaze_x_location / screen_width
        gaze_y_location = gaze_y_location / screen_height

    return gaze_x_location, gaze_y_location
    #
    # x_location = x_offset - screen_origin_x
    # y_location = y_offset - screen_origin_y
    #
    # if normalized:
    #     x_location = screen_width / x_location
    #     y_location = screen_height / y_location
    #
    # return x_location, y_location
    pass

# Read the CSV file
gaze_info = pd.read_csv('GazeInfo.csv')

timestamps = gaze_info[['LocalClock']].values
fs = len(timestamps) / (timestamps[-1] - timestamps[0])

gaze_position = gaze_info[['GazePixelPositionX', 'GazePixelPositionY']]
gaze_position_data = gaze_position.values
# Display the contents of the DataFrame
print(gaze_position)

# test_rns = RNStream('D:/1/03_09_2023_15_55_02-Exp_RenaPipline-Sbj_zl-Ssn_0.p')
# test_reloaded_data = test_rns.stream_in(jitter_removal=False)

file_path = 'D:/1/03_09_2023_15_55_02-Exp_RenaPipline-Sbj_zl-Ssn_0.p'

with open(file_path, 'rb') as file:
    # Load the data from the pickle file
    data = pickle.load(file)

eye_tracking_data = data['Unity.VarjoEyeTrackingComplete'][0]
eye_tracking_timestamp = data['Unity.VarjoEyeTrackingComplete'][1]

index_gaze_forward_x = varjo_channels.index("gaze_forward_x")
index_gaze_forward_y = varjo_channels.index("gaze_forward_y")
index_gaze_forward_z = varjo_channels.index("gaze_forward_z")

index_left_pupil_size = varjo_channels.index("left_pupil_size")
index_right_pupil_size = varjo_channels.index("right_pupil_size")


gaze_forward_x = data['Unity.VarjoEyeTrackingComplete'][0][index_gaze_forward_x, :]
gaze_forward_y = data['Unity.VarjoEyeTrackingComplete'][0][index_gaze_forward_y, :]
gaze_forward_z = data['Unity.VarjoEyeTrackingComplete'][0][index_gaze_forward_z, :]
left_pupil_size = data['Unity.VarjoEyeTrackingComplete'][0][index_left_pupil_size, :]
right_pupil_size = data['Unity.VarjoEyeTrackingComplete'][0][index_right_pupil_size, :]


gaze_locations = []
for index, timestamps in enumerate(eye_tracking_timestamp):
    gaze_location = gaze_vector_to_screen_space(gaze_vector=(gaze_forward_x[index],gaze_forward_y[index],gaze_forward_z[index]),
                                                gaze_origin=(0.5,0.5,1), screen_width=1, screen_height=1)
    gaze_locations.append(gaze_location)

gaze_locations = np.array(gaze_locations)

left_pupil_size = left_pupil_size[:, np.newaxis]
right_pupil_size = right_pupil_size[:, np.newaxis]

gaze_data = np.concatenate((gaze_locations, left_pupil_size, right_pupil_size), axis=1)

with open('eyelink_1000_dummy.p', 'wb') as file:
    pickle.dump(gaze_data, file)

print(gaze_locations)




print("John")
