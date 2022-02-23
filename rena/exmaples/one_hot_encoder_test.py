import numpy as np
import json
from utils.data_utils import RNStream, integer_one_hot
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

exp_info_dict = json.load(open('../utils/IndexPen_utils/IndexPenExp.json'))




# get useful timestamps
DataStreamName = 'TImmWave_6843AOP'

ExpID = exp_info_dict['ExpID']
ExpLSLStreamName = exp_info_dict['ExpLSLStreamName']
ExpStartMarker = exp_info_dict['ExpStartMarker']
ExpEndMarker = exp_info_dict['ExpEndMarker']
ExpLabelMarker = exp_info_dict['ExpLabelMarker']
ExpInterruptMarker = exp_info_dict['ExpInterruptMarker']
ExpErrorMarker = exp_info_dict['ExpErrorMarker']

category = list(ExpLabelMarker.values())
encoder = OneHotEncoder(categories='auto')
encoder.fit(np.reshape(category, (-1, 1)))

y = [1, 2, 3, 4, 5, 6]
test = encoder.transform(np.reshape(y, (-1, 1))).toarray()


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

import tensorflow as tf
print("GPUs: ", len(tf.config.experimental.list_physical_devices('GPU')))