# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from eidl.utils.model_utils import load_image_preprocess, get_trained_model
#
# # from source.utils.model_utils import get_trained_model, load_image
#
# # replace the image path to yours
# image_path = r'D:\HaowenWei\PycharmProjects\PhysioLabXR\physiolabxr\scripting\AOIAugmentationScript\test\edge_detector\test_image.png'
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device)
#
# image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)
#
# # show the image
#
# plt.imshow(image)
#
# # get the prediction
# y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device), collapse_attention_matrix=False)
# predicted_label = np.array([torch.argmax(y_pred).item()])
# decoded_label = compound_label_encoder.decode(predicted_label)
#
# # note: average over the column. Axis = 1
#
# print(f'Predicted label: {decoded_label}')


import numpy as np
import matplotlib.pyplot as plt
import torch
import matplotlib
from eidl.utils.model_utils import get_trained_model, load_image_preprocess
import cv2

# replace the image path to yours
image_path = r'/physiolabxr/scripting/illumiRead/AOIAugmentationScript/test/edge_detector/images/02_8981_OS_2021_widefield_report.png'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model, image_mean, image_std, image_size, compound_label_encoder = get_trained_model(device, model_param='num-patch-32_image-size-1024-512')

image_normalized, image = load_image_preprocess(image_path, image_size, image_mean, image_std)


# get the prediction
y_pred, attention_matrix = model(torch.Tensor(image_normalized).unsqueeze(0).to(device), collapse_attention_matrix=False)
predicted_label = np.array([torch.argmax(y_pred).item()])
decoded_label = compound_label_encoder.decode(predicted_label)

print(f'Predicted label: {decoded_label}')

attention_matrix = attention_matrix.squeeze().cpu().detach().numpy()
attention_matrix_without_class_token = attention_matrix[1:, 1:]


average_attention = np.mean(attention_matrix_without_class_token, axis=1)
average_attention_grid = average_attention.reshape(32, 32)
average_attention_grid = (average_attention_grid - np.min(average_attention_grid)) / (np.max(average_attention_grid) - np.min(average_attention_grid))


# Upsample factors
attention_grid_upscale_x = 1024 // 32
attention_grid_upscale_y = 512 // 32

# Upsample the image using NumPy's repeat function
attention_image = np.repeat(np.repeat(average_attention_grid, attention_grid_upscale_x, axis=1), attention_grid_upscale_y, axis=0)

# minimax normalization
# attention_image = (attention_image - np.min(attention_image)) / (np.max(attention_image) - np.min(attention_image))

cmap = plt.get_cmap('hot')
attention_image_heatmap = cmap(attention_image)
plt.imshow(attention_image_heatmap)
plt.show()


th = np.where(attention_image > 0.85, 1, 0)
plt.imshow(th)
plt.show()


