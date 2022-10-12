import numpy as np
import cv2
import torch


def prepare_image_for_sim_score(img):
    img = np.moveaxis(cv2.resize(img, (64, 64)), -1, 0)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    return img

def add_bounding_box(a, x, y, width, height, color):
    copy = np.copy(a)
    image_height = a.shape[0]
    image_width = a.shape[1]
    bounding_box = (np.max([0, x - int(width/2)]), np.min([np.max([0, y - int(height/2)]), a.shape[1]-1]), width, height)

    copy[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    copy[np.min([image_height-1, bounding_box[1] + bounding_box[3]]), bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], np.min([image_width-1, bounding_box[0] + bounding_box[2]])] = color
    return copy
