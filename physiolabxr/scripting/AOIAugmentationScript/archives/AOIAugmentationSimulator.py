# import numpy as np
#
# gaze_data = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

from rena.scripting.AOIAugmentationScript.AOIAugmentationUtils import GazeAttentionMatrixTorch

if __name__ == '__main__':
    device = torch.device('cuda:0')

    image_shape = np.array([500, 1000])
    attention_grid_shape = np.array([25, 50])
    attention_patch_shape = np.array([20,20])
    a = GazeAttentionMatrixTorch(image_shape=image_shape, attention_patch_shape=attention_patch_shape, device=device)
    a.add_attention(attention_center_location=[100, 100])
    a.decay()
    a.convolve_attention_grid_buffer()

    while 1:
        attention_add_start = time.perf_counter_ns()
        a.add_attention(attention_center_location=[100, 100])
        attention_add_time = time.perf_counter_ns()-attention_add_start

        attention_decay_start = time.perf_counter_ns()
        a.decay()
        attention_decay_time = time.perf_counter_ns()-attention_decay_start


        attention_grid_average_start = time.perf_counter_ns()
        a.convolve_attention_grid_buffer()
        attention_grid_average_time = time.perf_counter_ns()-attention_grid_average_start

        detach_start = time.perf_counter_ns()
        b = a.get_attention_grid_buffer.view(25, 50).cpu()
        detach_time = time.perf_counter_ns() - detach_start

        print(attention_add_time*1e-6, attention_decay_time*1e-6, attention_grid_average_time*1e-6, detach_time*1e-6)
        print('time cost:', (attention_add_time+attention_decay_time+attention_grid_average_time+detach_time)*1e-6)

