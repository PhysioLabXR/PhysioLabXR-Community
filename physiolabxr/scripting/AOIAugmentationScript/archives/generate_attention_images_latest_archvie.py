import os
import pickle

import cv2
import matplotlib.pyplot as plt
# from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
import numpy as np
import torch
from eidl.utils.model_utils import get_trained_model, load_image_preprocess
from eidl.viz.vit_rollout import VITAttentionRollout


class ImageInfo():
    def __init__(self,image_path, image, image_normalized, attention_patch_shape, raw_attention_matrix,
                 rollout_attention_matrix, average_self_attention_matrix,
                 y=None, y_pred=None):

        self.image_path = image_path
        self.image = image
        self.image_normalized = image_normalized
        self.attention_patch_shape = attention_patch_shape
        self.attention_grid_shape = (image.shape[0] // attention_patch_shape[0], image.shape[1] // attention_patch_shape[1])
        self.raw_attention_matrix = raw_attention_matrix

        self.y = y
        self.y_pred = y_pred

def overlay_heatmap(original, heatmap, image_size, alpha=.5, interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_JET, normalize=False):
    if normalize:
        heatmap = heatmap / heatmap.max()
    heatmap_processed = cv2.resize(heatmap, dsize=image_size, interpolation=interpolation)
    heatmap_processed = cv2.applyColorMap(cv2.cvtColor((heatmap_processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), cmap)
    heatmap_processed = cv2.cvtColor(heatmap_processed, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original.astype(np.uint8), alpha, heatmap_processed, 1 - alpha, 0), heatmap_processed


##########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                   model_param='num-patch-32_image-size-1024-512')
##########################################################################


# root_dir = r'D:\HaowenWei\UnityProject\PerceptualAOIAugmentation\Assets\Prefabs\ExperimentImages\Practice'
root_dir = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\reports_cleaned\S'
target_dir = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\data\report_cleand_attention_vis\S'

y_true = 'S'

image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]


data_dict = {}

for index, image_name in enumerate(image_names):

    image_path = os.path.join(root_dir, image_name)

    # get the prediction and attention matrix
    image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std) # the normalized image is z normalization

    image_tensor = torch.Tensor(image_normalized).unsqueeze(0).to(device)
    y_pred, attention_matrix = model(image_tensor,
                                     collapse_attention_matrix=False)

    predicted_label = np.array([torch.argmax(y_pred).item()])
    print(f'y_pred: {y_pred}')
    decoded_label = compound_label_encoder.decode(predicted_label)
    print(f'Predicted label: {decoded_label}')

    # plt.imshow(image.astype(np.uint8))
    # plt.title(f'y_true: {[y_true]}, y_pred: {decoded_label}')
    # plt.colorbar()
    # plt.show()

    vit_rollout = VITAttentionRollout(model, device=device, attention_layer_name='attn_drop', head_fusion="mean",
                                      discard_ratio=0.5)
    rollout = vit_rollout(depth=model.depth, input_tensor=image_tensor)

    rollout_overlap, rollout_heatmap  = overlay_heatmap(image, rollout, image_size, alpha=0.5, interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_VIRIDIS)
    attention_matrix_self_attention = attention_matrix.squeeze().detach().cpu().numpy()[1:, 1:]
    attention_matrix_self_attention = np.mean(attention_matrix_self_attention, axis=0).reshape(model.grid_size)
    self_attention_overlap, self_attention_heatmap = overlay_heatmap(image, attention_matrix_self_attention, image_size, alpha=0.3, interpolation=cv2.INTER_NEAREST, normalize=True, cmap=cv2.COLORMAP_VIRIDIS)

    fig = plt.figure(figsize=(30, 45), constrained_layout=True)
    axes = fig.subplots(3, 2)
    axes[0, 0].imshow(image.astype(np.uint8))  # plot the original image
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f'Original image. y_true: {[y_true]}, y_pred: {decoded_label}', fontsize = 40)

    axes[1, 0].imshow(rollout_heatmap)  # plot the attention rollout
    axes[1, 0].axis('off')
    axes[1, 0].set_title(f'Attention rollout', fontsize = 40)

    axes[2, 0].imshow(rollout_overlap)  # plot the attention rollout
    axes[2, 0].axis('off')
    axes[2, 0].set_title(f'Overlayed attention rollout', fontsize = 40)

    axes[1, 1].imshow(self_attention_heatmap)  # plot the attention rollout
    axes[1, 1].axis('off')
    axes[1, 1].set_title(f'Self Attention', fontsize = 40)

    axes[2, 1].imshow(self_attention_overlap)  # plot the attention rollout
    axes[2, 1].axis('off')
    axes[2, 1].set_title(f'Overlayed Self Attention', fontsize = 40)

    # plt.show()
    plt.savefig(os.path.join(target_dir, image_name), dpi=500)





