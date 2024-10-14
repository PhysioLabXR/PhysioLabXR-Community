import numpy as np
import torch
from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import GazeAttentionMatrix
import matplotlib.pyplot as plt
import pickle

def fix_seq_to_gaze_map(fix_seq, original_image_size, max_image_size=(3000, 6000)):
    current_image_shape = np.array(original_image_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gaze_attention_matrix = GazeAttentionMatrix(device=device)

    gaze_attention_matrix.set_maximum_image_shape(np.array(max_image_size))
    gaze_attention_matrix.gaze_attention_pixel_map_buffer = torch.tensor(np.zeros(shape=current_image_shape), device=device)

    for fixation_point in fix_seq:
        gaze_on_image_attention_map = gaze_attention_matrix.get_gaze_on_image_attention_map(fixation_point, current_image_shape)
        gaze_attention_matrix.gaze_attention_pixel_map_clutter_removal(gaze_on_image_attention_map, attention_clutter_ratio=0.995)
    gaze_attention_map = gaze_attention_matrix.gaze_attention_pixel_map_buffer.detach().cpu().numpy()
    return gaze_attention_map


if __name__ == '__main__':
    image_info = pickle.load(open( r'C:\PycharmProjects\temp\perceptual_roi\samples\9071_OD_2021_widefield_report_01_18_2024_13_41_13_source-attention-info.p', 'rb'))

    image_name = image_info['image_name']
    gaze_attention_map = image_info['gaze_attention_map']
    fixation_sequence = image_info['fixation_sequence']

    gaze_attention_map = fix_seq_to_gaze_map(fixation_sequence, gaze_attention_map.shape)

    plt.imshow(gaze_attention_map)
    plt.show()





