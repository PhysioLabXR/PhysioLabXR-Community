import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageInfo():
    def __init__(self, image_path, original_image,
                 image_to_model, image_to_model_normalized, model_image_shape,
                 patch_shape, attention_grid_shape,
                 raw_attention_matrix,
                 rollout_attention_matrix, average_self_attention_matrix,
                 y=None, y_pred=None):
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
        self.y = y
        self.y_pred = y_pred



read_pickle_file = 'data_dict.pkl'

with open(read_pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

print(data_dict.keys())


test_image_info = data_dict['G']['9172_OD_2021_widefield_report.png']
plt.imshow(test_image_info.image_to_model.astype(np.uint8))
plt.show()

plt.imshow(test_image_info.rollout_attention_matrix)
plt.show()

rollout_attention_matrix = test_image_info.rollout_attention_matrix
rollout_attention_matrix = cv2.resize(rollout_attention_matrix, (test_image_info.original_image.shape[1], test_image_info.original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

ret, thresh = cv2.threshold(rollout_attention_matrix, 0.5, 1, 0)
plt.imshow(thresh)
plt.show()


contours, hierarchy = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours on original image with red color and thickness 20
image = test_image_info.original_image.astype(np.uint8)

# bgr to rgb
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image = cv2.drawContours(image, contours, -1, (0, 0, 255), 20)

# plot image with high resolution
plt.figure(figsize=(30, 15))
plt.imshow(image)
plt.show()

