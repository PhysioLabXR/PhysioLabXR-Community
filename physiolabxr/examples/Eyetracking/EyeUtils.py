import numpy as np
import cv2
import torch


def prepare_image_for_sim_score(img):
    img = np.moveaxis(cv2.resize(img, (64, 64)), -1, 0)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    return img

def add_bounding_box(a, x, y, width, height, color):
    """Add a bounding box to an image.

    Args:
        a (np.ndarray): Image to add the bounding box to.
        x (int): x-coordinate of the center of the bounding box.
        y (int): y-coordinate of the center of the bounding box.
        width (int): Width of the bounding box.
        height (int): Height of the bounding box.
        color (tuple): Color of the bounding box.
    """
    copy = np.copy(a)
    image_height = a.shape[0]
    image_width = a.shape[1]
    bounding_box = (np.max([0, x - int(width/2)]), np.min([np.max([0, y - int(height/2)]), a.shape[1]-1]), width, height)

    copy[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    copy[np.min([image_height-1, bounding_box[1] + bounding_box[3]]), bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], np.min([image_width-1, bounding_box[0] + bounding_box[2]])] = color
    return copy

    # # Create a copy of the image to avoid modifying the original
    # copy = np.copy(a)
    # image_height, image_width = a.shape[:2]
    #
    # # Calculate the top-left corner of the bounding box
    # x1 = max(0, x - width // 2)
    # y1 = max(0, y - height // 2)
    # x2 = min(image_width - 1, x + width // 2)
    # y2 = min(image_height - 1, y + height // 2)
    #
    # # Draw the bounding box
    # copy[y1:y2, x1] = color  # Left vertical line
    # copy[y1:y2, x2] = color  # Right vertical line
    # copy[y1, x1:x2] = color  # Top horizontal line
    # copy[y2, x1:x2] = color  # Bottom horizontal line
    #
    # return copy

def clip_bbox(x_center, y_center, width, height, image_shape):
    H, W = image_shape[:2]  # Get the height and width of the image

    # Calculate the top-left and bottom-right corners
    x1 = min(max(0, x_center - width // 2), W-1)
    y1 = min(max(0, y_center - height // 2), H-1)
    x2 = max(min(W-1, x_center + width // 2), 0)
    y2 = max(min(H-1, y_center + height // 2), 0)

    # Recalculate the width and height after clipping
    clipped_width = x2 - x1
    clipped_height = y2 - y1

    # Return the clipped bounding box in the original format
    x_center_clipped = x1 + clipped_width // 2
    y_center_clipped = y1 + clipped_height // 2

    return (x_center_clipped, y_center_clipped, clipped_width, clipped_height)