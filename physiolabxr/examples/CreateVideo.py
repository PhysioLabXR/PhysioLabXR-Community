import os

import cv2

from physiolabxr.configs.configs import AppConfigs
from physiolabxr.utils.RNStream import RNStream

data_root = 'C:/Users/S-Vec/Dropbox/research/RealityNavigation/Data/Pilot/'
data_fn = '03_22_2021_17_03_52-Exp_realitynavigation-Sbj_0-Ssn_2.dats'

video_stream_label = 'monitor1'

rns = RNStream(os.path.join(data_root, data_fn))
data = rns.stream_in(ignore_stream=('0'))

video_frame_stream = data[video_stream_label][0]
frame_count = video_frame_stream.shape[-1]
frame_size = (data[video_stream_label][0].shape[1], data[video_stream_label][0].shape[0])
out_path = os.path.join(data_root, '{0}_{1}.avi'.format(data_fn.split('.')[0], video_stream_label))
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 1 / AppConfigs().video_device_refresh_interval, frame_size)

for i in range(frame_count):
    print('Creating video progress {}%'.format(str(round(100 * i / frame_count, 2))), sep=' ', end='\r',
          flush=True)
    img = video_frame_stream[:, :, :, i]
    # img = np.reshape(img, newshape=list(frame_size) + [-1,])
    out.write(img)

out.release()
