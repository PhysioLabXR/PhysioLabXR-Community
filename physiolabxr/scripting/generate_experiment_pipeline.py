import numpy as np
import random

report_list = ['8898_OD_2021_widefield_report', '8902_OS_2021_widefield_report', '8914_OD_2021_widefield_report', '8918_OS_2021_widefield_report', '8919_OD_2021_widefield_report', '8924_OS_2021_widefield_report', '8935_OD_2021_widefield_report', '8939_OS_2021_widefield_report', '8974_OD_2021_widefield_report', '8981_OS_2021_widefield_report', '8984_OD_2021_widefield_report', '8988_OS_2021_widefield_report', '9000_OD_2021_widefield_report', '9002_OS_2021_widefield_report', '9025_OD_2021_widefield_report', '9071_OD_2021_widefield_report', '9075_OD_2021_widefield_report', '9078_OS_2021_widefield_report', '9108_OD_2021_widefield_report', '9111_OS_2021_widefield_report', '9140_OD_2021_widefield_report', '9172_OD_2021_widefield_report', '9175_OS_2021_widefield_report', '9183_OD_2021_widefield_report', '9184_OS_2021_widefield_report', '9238_OD_2021_widefield_report', 'RLS_001_OD_TC', 'RLS_001_OS_TC', 'RLS_004_OD_TC', 'RLS_004_OS_TC', 'RLS_005_OD_TC', 'RLS_005_OS_TC', 'RLS_006_OD_TC', 'RLS_008_OD_TC', 'RLS_008_OS_TC', 'RLS_011_OD_TC', 'RLS_011_OS_TC', 'RLS_012_OS_TC', 'RLS_016_OD_TC', 'RLS_018_OD_TC', 'RLS_018_OS_TC', 'RLS_023_OD_TC', 'RLS_023_OS_TC', 'RLS_024_OD_TC', 'RLS_024_OS_TC', 'RLS_025_OD_TC', 'RLS_028_OS_TC', 'RLS_031_OD_TC', 'RLS_031_OS_TC', 'RLS_033_OS_TC', 'RLS_036_OD_TC', 'RLS_036_OS_TC', 'RLS_039_OD_TC', 'RLS_039_OS_TC', 'RLS_040_OD_TC', 'RLS_041_OD_TC', 'RLS_042_OD_TC', 'RLS_043_OD_TC', 'RLS_043_OS_TC', 'RLS_044_OS_TC', 'RLS_045_OD_TC', 'RLS_045_OS_TC', 'RLS_046_OS_TC', 'RLS_048_OD_TC', 'RLS_048_OS_TC', 'RLS_050_OS_TC', 'RLS_054_OD_TC', 'RLS_054_OS_TC', 'RLS_058_OD_TC', 'RLS_059_OS_TC', 'RLS_060_OD_TC', 'RLS_060_OS_TC', 'RLS_062_OD_TC', 'RLS_062_OS_TC', 'RLS_063_OD_TC', 'RLS_067_OD_TC', 'RLS_068_OD_TC', 'RLS_068_OS_TC', 'RLS_073_OD_TC', 'RLS_073_OS_TC', 'RLS_077_OD_TC', 'RLS_077_OS_TC', 'RLS_079_OD_TC', 'RLS_079_OS_TC', 'RLS_082_OD_TC', 'RLS_082_OS_TC', 'RLS_083_OD_TC', 'RLS_083_OS_TC', 'RLS_084_OD_TC', 'RLS_084_OS_TC', 'RLS_085_OD_TC', 'RLS_086_OD_TC', 'RLS_086_OS_TC', 'RLS_087_OD_TC', 'RLS_087_OS_TC', 'RLS_089_OS_TC', 'RLS_091_OD_TC', 'RLS_091_OS_TC', 'RLS_093_OD_TC', 'RLS_093_OS_TC', 'RLS_096_OD_TC', 'RLS_096_OS_TC', 'RLS_097_OD_TC', 'RLS_097_OS_TC', 'RLS_074_OD_TC', 'RLS_074_OS_TC', 'RLS_076_OD_TC', 'RLS_076_OS_TC', 'RLS_078_OD_TC', 'RLS_078_OS_TC', 'RLS_080_OD_TC', 'RLS_080_OS_TC', 'RLS_081_OD_TC', 'RLS_081_OS_TC', 'RLS_085_OS_TC', 'RLS_092_OD_TC', 'RLS_092_OS_TC', 'RLS_095_OD_TC', 'RLS_095_OS_TC', '8904_OD_2021_widefield_report', '8909_OS_2021_widefield_report', '8910_OD_2021_widefield_report', '8960_OD_2021_widefield_report', '8962_OS_2021_widefield_report', '9059_OD_2021_widefield_report', '9061_OS_2021_widefield_report', '9084_OS_2021_widefield_report', '9103_OD_2021_widefield_report', '9105_OS_2021_widefield_report', '9187_OD_2021_widefield_report', '9189_OS_2021_widefield_report', '9191_OD_2021_widefield_report', '9193_OS_2021_widefield_report', '9219_OD_2021_widefield_report', '9222_OS_2021_widefield_report', '9223_OD_2021_widefield_report', '9226_OS_2021_widefield_report', '9257_OD_2021_widefield_report', '9261_OS_2021_widefield_report', 'RLS_002_OD_TC', 'RLS_002_OS_TC', 'RLS_009_OD_TC', 'RLS_009_OS_TC', 'RLS_014_OD_TC', 'RLS_014_OS_TC', 'RLS_019_OD_TC', 'RLS_019_OS_TC', 'RLS_038_OD_TC', 'RLS_047_OD_TC', 'RLS_047_OS_TC', 'RLS_051_OD_TC', 'RLS_051_OS_TC', 'RLS_053_OD_TC', 'RLS_053_OS_TC', 'RLS_057_OD_TC', 'RLS_057_OS_TC', 'RLS_064_OD_TC', 'RLS_064_OS_TC', 'RLS_113_OD_TC', 'RLS_113_OS_TC', 'RLS_125_OD_TC', 'RLS_125_OS_TC', 'RLS_127_OD_TC', 'RLS_127_OS_TC', 'RLS_148_OD_TC', 'RLS_148_OS_TC', 'RLS_150_OD_TC', 'RLS_150_OS_TC', '8942_OD_2021_widefield_report', '9003_OD_2021_widefield_report', '9005_OS_2021_widefield_report', '9014_OD_2021_widefield_report', '9016_OS_2021_widefield_report', '9194_OD_2021_widefield_report', '9197_OS_2021_widefield_report', '9198_OD_2021_widefield_report', '9201_OS_2021_widefield_report']

participant_id = 0

g_list = report_list[0:16]
s_list = report_list[-17:-1]

image_list = g_list+s_list
image_condition_list = [1,2,3,4,4,3,2,1,1,2,3,4,4,3,2,1,1,2,3,4,4,3,2,1,1,2,3,4,4,3,2,1]

image_condition_list = np.roll(image_condition_list, 1)

combined_list = list(zip(image_list, image_condition_list))

shuffled_list1, shuffled_list2 = zip(*combined_list)





