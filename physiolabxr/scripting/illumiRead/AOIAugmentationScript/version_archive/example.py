# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# from physiolabxr.scripting.AOIAugmentationScript.AOIAugmentationConfig import *
# import matplotlib.pyplot as plt
#
# def generate_image_binary_mask(image, depth_first=False):
#     if depth_first:
#         image = np.moveaxis(image, 0, -1)
#
#     # Convert the RGB image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary_mask = cv2.threshold(gray_image, 254, 1, cv2.THRESH_BINARY_INV)
#     return binary_mask
#
# def generate_attention_grid_mask(image_mask, attention_patch_shape):
#     kernel = torch.tensor(np.ones(shape=(attention_patch_shape[0], attention_patch_shape[1])), dtype=torch.float32)
#     image_mask = torch.tensor(image_mask, dtype=torch.float32)
#
#     attention_grid_mask = F.conv2d(input=image_mask.view(1, 1, image_mask.shape[0], image_mask.shape[1]),
#                                    weight=kernel.view(1, 1, attention_patch_shape[0], attention_patch_shape[1]),
#                                    stride=(attention_patch_shape[0], attention_patch_shape[1]))
#
#     attention_grid_mask = attention_grid_mask.squeeze().cpu().numpy()
#     attention_grid_mask = np.where(attention_grid_mask > 0, 1, 0)
#     return attention_grid_mask
#
#
# def generate_square_attention_matrix_mask(attention_grid_mask):
#     attention_grid_mask_flatten = attention_grid_mask.flatten()
#     attention_matrix_mask = np.ones(shape=(attention_grid_mask_flatten.shape[0], attention_grid_mask_flatten.shape[0]))
#
#     for i in range(attention_grid_mask_flatten.shape[0]):
#         if attention_grid_mask_flatten[i] == 0:
#             attention_matrix_mask[i, :] = 0
#             attention_matrix_mask[:, i] = 0
#
#     return attention_matrix_mask
#
#
# def get_attention_matrix(image_path, image_shape, attention_patch_shape, mask_white=True):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (image_shape[1], image_shape[0]))
#     attention_grid_shape = (image_shape[0] // attention_patch_shape[0], image_shape[1] // attention_patch_shape[1])
#     attention_matrix = np.random.rand(attention_grid_shape[0]*attention_grid_shape[1], attention_grid_shape[0]*attention_grid_shape[1])
#
#     if mask_white:
#         binary_mask = generate_image_binary_mask(image)
#         attention_grid_mask = generate_attention_grid_mask(binary_mask, attention_patch_shape=attention_patch_shape)
#         attention_matrix_mask = generate_square_attention_matrix_mask(attention_grid_mask)
#         attention_matrix = attention_matrix * attention_matrix_mask
#
#     return attention_matrix
#
#
#
# def get_attention_map_dict():
#     pass
#
#
# if __name__ == '__main__':
#     image_path = 'D:/HaowenWei/UnityProject/PerceptualAOIAugmentation/Assets/Prefabs/OCTReportImages/Test/1.png'
#     attention_matrix = get_attention_matrix(image_path=image_path, image_shape=image_shape,attention_patch_shape=attention_patch_shape ,mask_white=True)