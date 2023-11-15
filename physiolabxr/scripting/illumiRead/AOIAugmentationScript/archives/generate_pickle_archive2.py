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
    def __init__(self,image_path, image, image_normalized, attention_patch_shape, attention_matrix, y=None, y_pred=None):
        self.image_path = image_path
        self.image = image
        self.image_normalized = image_normalized
        self.attention_patch_shape = attention_patch_shape
        self.attention_matrix = attention_matrix
        self.y = y
        self.y_pred = y_pred






##########################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device,
                                                                                   model_param='num-patch-32_image-size-1024-512')
##########################################################################


root_dir = r'D:\HaowenWei\UnityProject\PerceptualAOIAugmentation\Assets\Prefabs\ExperimentImages\Practice'
image_names = [file for file in os.listdir(root_dir) if file.endswith('.png') or file.endswith('.jpg')]


data_dict = {}

y_true = 'G'

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

    rollout_resized = cv2.resize(rollout, dsize=image_size, interpolation=cv2.INTER_LINEAR)
    rollout_heatmap = cv2.applyColorMap(cv2.cvtColor((rollout_resized * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                                        cv2.COLORMAP_JET)
    rollout_heatmap = cv2.cvtColor(rollout_heatmap, cv2.COLOR_BGR2RGB)
    alpha = 0.2
    output_image = cv2.addWeighted(image.astype(np.uint8), alpha, rollout_heatmap, 1 - alpha, 0)

    fig = plt.figure(figsize=(15, 30), constrained_layout=True)
    axes = fig.subplots(3, 1)
    axes[0].imshow(image.astype(np.uint8))  # plot the original image
    axes[0].axis('off')
    axes[0].set_title(f'Original image y_pred: {decoded_label}')

    axes[1].imshow(rollout_heatmap)  # plot the attention rollout
    axes[1].axis('off')
    axes[1].set_title(f'Attention rollout')

    axes[2].imshow(output_image)  # plot the attention rollout
    axes[2].axis('off')
    axes[2].set_title(f'Overlayed attention rollout')
    plt.show()

#     # attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()
#     #
#     # class_token_attention = attention_matrix[0, 1:]
#     # # class_token_attention = attention_matrix[1:, 0]
#     #
#     # attention_grid = class_token_attention.reshape(AOIAugmentationConfig.attention_grid_shape)
#     #
#     # attention_grid_upsample = np.repeat(attention_grid,2, axis=1)
#     # plt.imshow(attention_grid_upsample)
#     # plt.colorbar()
#     # plt.show()
#     #
#     #
#     #
#     #
#     # image_attention_info = ImageInfo(
#     #     image_path=image_path,
#     #     image=image,
#     #     image_normalized=image_normalized,
#     #     attention_patch_shape=AOIAugmentationConfig.attention_patch_shape,
#     #     attention_matrix=attention_matrix,
#     #     y=y_true,
#     #     y_pred=decoded_label
#     # )
#
#
# with open('../data/experiment_image_info/practice', 'wb') as file:
#     # A new file will be created
#     pickle.dump(data_dict, file)
