from physiolabxr.scripting.illumiRead.AOIAugmentationScript.AOIAugmentationUtils import gaussian_filter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


file_path = 'images/Sample 0 in test set, original image.png'
image = cv2.imread('images/Sample 0 in test set, original image.png')


def overlay_heatmap(original, heatmap, image_size, alpha=.5, interpolation=cv2.INTER_NEAREST, cmap=cv2.COLORMAP_JET, normalize=False):
    if normalize:
        heatmap = heatmap / heatmap.max()
        # min max normalization
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap_processed = cv2.resize(heatmap, dsize=image_size, interpolation=interpolation)
    heatmap_processed = cv2.applyColorMap(cv2.cvtColor((heatmap_processed * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), cmap)
    heatmap_processed = cv2.cvtColor(heatmap_processed, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original.astype(np.uint8), alpha, heatmap_processed, 1 - alpha, 0), heatmap_processed


center = np.array([861,2893]) # 0
# center = np.array([699,2685]) # 0
# center = np.array([776,3994]) # 0




gaze_attention = gaussian_filter(image.shape[0:2], center, sigma=40.0, normalized=True)
plt.imshow(gaze_attention)
plt.show()

attention_original_overlay, attention_image = overlay_heatmap(original=image,
                                                              heatmap=gaze_attention,
                                                              image_size = (image.shape[1], image.shape[0]), # (width, height)
                                                              alpha=.5,
                                                              interpolation=cv2.INTER_NEAREST,
                                                              cmap=cv2.COLORMAP_JET,
                                                              normalize=False)

# plt.imshow(attention_original_overlay)
# plt.show()


fig = plt.figure(figsize=(40, 15), constrained_layout=True)
axes = fig.subplots(1, 2)

axes[0].imshow(attention_original_overlay)  # plot the original image
axes[0].axis('off')
axes[0].set_title(f'Original image Overlay Overlap', fontsize=50)
#
axes[1].imshow(attention_image)  # plot the normalized image
axes[1].axis('off')
axes[1].set_title(f'Attention Map', fontsize=50)

plt.show()


with open(file_path, 'wb') as file:
    # Dump the data into the file
    pickle.dump(gaze_attention, file)