import os
import pickle

import cv2
import matplotlib.pyplot as plt
# from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationUtils import *
import numpy as np



class ImageInfo():
    def __init__(self, image_path, original_image,
                 image_to_model, image_to_model_normalized, model_image_shape,
                 patch_shape, attention_grid_shape,
                 raw_attention_matrix,
                 rollout_attention_matrix, average_self_attention_matrix,
                 y_true=None, y_pred=None):
        self.image_path = image_path
        self.original_image = original_image
        self.image_to_model = image_to_model
        self.image_to_model_normalized = image_to_model_normalized
        self.model_image_shape = model_image_shape
        self.patch_shape = patch_shape
        self.attention_grid_shape = attention_grid_shape
        self.raw_attention_matrix = raw_attention_matrix
        self.rollout_attention_matrix = rollout_attention_matrix
        self.average_self_attention_matrix = average_self_attention_matrix
        self.y_true = y_true
        self.y_pred = y_pred


def overlay_heatmap(original, heatmap, image_size, alpha=.5, interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_JET, normalize=False):
    if normalize:
        heatmap = heatmap / heatmap.max()
        # min max normalization
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_processed = cv2.resize(heatmap, dsize=image_size, interpolation=interpolation)
    heatmap_processed = cv2.applyColorMap(cv2.cvtColor((heatmap_processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), cmap)
    heatmap_processed = cv2.cvtColor(heatmap_processed, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original.astype(np.uint8), alpha, heatmap_processed, 1 - alpha, 0), heatmap_processed


def overlay_contour(original, heatmap, image_size, threshold=0.5, interpolation=cv2.INTER_NEAREST, color=(0, 0, 255)):
    # resize the heatmap to the original image size
    heatmap_processed = cv2.resize(heatmap, dsize=image_size, interpolation=interpolation)
    ret, thresh = cv2.threshold(heatmap_processed, threshold, 1, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    countour_image = cv2.drawContours(original.astype(np.uint8), contours, -1, color, 5)

    return countour_image, thresh, contours




target_dir = r'/physiolabxr/scripting/illumiRead/AOIAugmentationScript/data/report_cleaned_attention_vis'



data_dict_path = r'report_cleaned_image_info.pkl'
with open(data_dict_path, 'rb') as f:
    data_dict = pickle.load(f)

for class_name in data_dict:
    # if os.path.exists(os.path.join(target_dir, class_name)):
    #     os.remove(os.path.join(target_dir, class_name))

    os.mkdir(os.path.join(target_dir, class_name))

    class_dict = data_dict[class_name]
    for image_name in class_dict:
        image_info_dict = class_dict[image_name]
        image_info = ImageInfo(**image_info_dict)

        image = image_info.image_to_model
        image_normalized = image_info.image_to_model_normalized
        image_normalized = np.moveaxis(image_normalized, 0, -1)

        image_shape = image_info.model_image_shape
        rollout = image_info.rollout_attention_matrix
        attention_matrix_self_attention = image_info.average_self_attention_matrix

        y_true = image_info.y_true
        y_pred = image_info.y_pred


        # note the image shape and image size are different, image_shape = (h, w), image_size = (w, h)
        rollout_overlap, rollout_heatmap = overlay_heatmap(image, rollout, (image_shape[1], image_shape[0]),
                                                           alpha=0.5,
                                                           interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_VIRIDIS)

        self_attention_overlap, self_attention_heatmap = overlay_heatmap(image, attention_matrix_self_attention,
                                                                         (image_shape[1], image_shape[0]),
                                                                         alpha=0.5,
                                                                         interpolation=cv2.INTER_NEAREST,
                                                                         normalize=True, cmap=cv2.COLORMAP_VIRIDIS)

        contour_image_rollout_attention, thresholded_heatmap_rollout_attention, contours_rollout_attention = overlay_contour(image, rollout, (image_shape[1], image_shape[0]), threshold=0.5)
        contour_image_self_attention, thresholded_heatmap_self_attention, contours_self_attention = overlay_contour(image, attention_matrix_self_attention, (image_shape[1], image_shape[0]), threshold=0.5)


        fig = plt.figure(figsize=(40, 70), constrained_layout=True)
        axes = fig.subplots(5, 2)

        axes[0, 0].imshow(image.astype(np.uint8))  # plot the original image
        axes[0, 0].axis('off')
        axes[0, 0].set_title(f'Original image. y_true: {y_true}, y_pred: {y_pred}', fontsize = 50)

        axes[0, 1].imshow(image_normalized)  # plot the normalized image
        axes[0, 1].axis('off')
        axes[0, 1].set_title(f'Normalized image (Z normalization)', fontsize = 50)

        axes[1, 0].imshow(rollout_heatmap)  # plot the attention rollout
        axes[1, 0].axis('off')
        axes[1, 0].set_title(f'Attention rollout', fontsize = 50)

        axes[2, 0].imshow(rollout_overlap)  # plot the attention rollout
        axes[2, 0].axis('off')
        axes[2, 0].set_title(f'Overlaid attention rollout', fontsize = 50)

        axes[1, 1].imshow(self_attention_heatmap)  # plot the attention rollout
        axes[1, 1].axis('off')
        axes[1, 1].set_title(f'Self Attention', fontsize = 50)

        axes[2, 1].imshow(self_attention_overlap)  # plot the attention rollout
        axes[2, 1].axis('off')
        axes[2, 1].set_title(f'Overlaid Self Attention', fontsize = 50)


        axes[3,0].imshow(thresholded_heatmap_rollout_attention)  # plot the attention rollout
        axes[3,0].axis('off')
        axes[3,0].set_title(f'Thresholded Rollout Attention, thresh: 0.5', fontsize = 50)


        axes[3,1].imshow(thresholded_heatmap_self_attention)  # plot the attention rollout
        axes[3,1].axis('off')
        axes[3,1].set_title(f'Thresholded Self Attention, thresh: 0.5', fontsize = 50)

        axes[4, 0].imshow(contour_image_rollout_attention)  # plot the attention rollout
        axes[4, 0].axis('off')
        axes[4, 0].set_title(f'Contour Rollout Attention', fontsize = 50)

        axes[4, 1].imshow(contour_image_self_attention)  # plot the attention rollout
        axes[4, 1].axis('off')
        axes[4, 1].set_title(f'Contour Self Attention', fontsize = 50)


        # plt.show()

        fig.savefig(os.path.join(target_dir, class_name, image_name), dpi=300)



        # save the figure
        # fig.savefig(os.path.join(target_dir, class_name, image_name), dpi=300)








# ret, thresh = cv2.threshold(rollout_attention_matrix, 0.5, 1, 0)
# plt.imshow(thresh)
# plt.show()
#
#
# contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# # draw contours on original image with red color and thickness 20
# image = test_image_info.original_image.astype(np.uint8)
#
# # bgr to rgb
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# image = cv2.drawContours(image, contours, -1, (0, 0, 255), 20)
#
# # plot image with high resolution
# plt.figure(figsize=(30, 15))
# plt.imshow(image)
# plt.show()



# for index, image_name in enumerate(image_names):
#
#     image_path = os.path.join(root_dir, image_name)
#
#     # get the prediction and attention matrix
#     image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std) # the normalized image is z normalization
#
#     image_tensor = torch.Tensor(image_normalized).unsqueeze(0).to(device)
#     y_pred, attention_matrix = model(image_tensor,
#                                      collapse_attention_matrix=False)
#
#     predicted_label = np.array([torch.argmax(y_pred).item()])
#     print(f'y_pred: {y_pred}')
#     decoded_label = compound_label_encoder.decode(predicted_label)
#     print(f'Predicted label: {decoded_label}')
#
#     # plt.imshow(image.astype(np.uint8))
#     # plt.title(f'y_true: {[y_true]}, y_pred: {decoded_label}')
#     # plt.colorbar()
#     # plt.show()
#
#     vit_rollout = VITAttentionRollout(model, device=device, attention_layer_name='attn_drop', head_fusion="mean",
#                                       discard_ratio=0.5)
#     rollout = vit_rollout(depth=model.depth, input_tensor=image_tensor)
#
#     rollout_overlap, rollout_heatmap  = overlay_heatmap(image, rollout, image_size, alpha=0.5, interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_VIRIDIS)
#     attention_matrix_self_attention = attention_matrix.squeeze().detach().cpu().numpy()[1:, 1:]
#     attention_matrix_self_attention = np.mean(attention_matrix_self_attention, axis=0).reshape(model.grid_size)
#     self_attention_overlap, self_attention_heatmap = overlay_heatmap(image, attention_matrix_self_attention, image_size, alpha=0.3, interpolation=cv2.INTER_NEAREST, normalize=True, cmap=cv2.COLORMAP_VIRIDIS)
#
#     fig = plt.figure(figsize=(30, 45), constrained_layout=True)
#     axes = fig.subplots(3, 2)
#     axes[0, 0].imshow(image.astype(np.uint8))  # plot the original image
#     axes[0, 0].axis('off')
#     axes[0, 0].set_title(f'Original image. y_true: {[y_true]}, y_pred: {decoded_label}', fontsize = 40)
#
#     axes[1, 0].imshow(rollout_heatmap)  # plot the attention rollout
#     axes[1, 0].axis('off')
#     axes[1, 0].set_title(f'Attention rollout', fontsize = 40)
#
#     axes[2, 0].imshow(rollout_overlap)  # plot the attention rollout
#     axes[2, 0].axis('off')
#     axes[2, 0].set_title(f'Overlayed attention rollout', fontsize = 40)
#
#     axes[1, 1].imshow(self_attention_heatmap)  # plot the attention rollout
#     axes[1, 1].axis('off')
#     axes[1, 1].set_title(f'Self Attention', fontsize = 40)
#
#     axes[2, 1].imshow(self_attention_overlap)  # plot the attention rollout
#     axes[2, 1].axis('off')
#     axes[2, 1].set_title(f'Overlayed Self Attention', fontsize = 40)
#
#     # plt.show()
#     plt.savefig(os.path.join(target_dir, image_name), dpi=500)





