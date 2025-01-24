"""
This class is responsible for loading the data and run inference on the data with SAM-2
"""
import json
import sys
import os
from typing import List, Dict
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.gather_raw_data_comp import ToTensor,get_target_sequence_dataloader

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#The total number of masks that are considered invalid
Rejected_cnt = 0

def compute_shift(img1, img2):
    """
    Compute the shift between two images using phase correlation
    Returns: (x_shift, y_shift)
    """
    # Convert images to grayscale if they're not already
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    # Compute phase correlation
    shift = cv2.phaseCorrelate(np.float32(img1_gray), np.float32(img2_gray))[0]
    return shift  # (x_shift, y_shift)

def align_sequence(frames):
    """
    Align all frames in a sequence to the first frame
    Returns: aligned frames and shifts
    """
    batch_size, seq_length = frames.shape[0], frames.shape[1]
    aligned_frames = frames.clone()
    all_shifts = []
    
    for b in range(batch_size):
        sequence_shifts = []
        reference_frame = frames[b, 0].permute(1, 2, 0).cpu().numpy()
        
        for s in range(1, seq_length):
            current_frame = frames[b, s].permute(1, 2, 0).cpu().numpy()
            shift = compute_shift(reference_frame, current_frame)
            sequence_shifts.append(shift)
            
            # Apply shift to frame
            M = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            aligned_frame = cv2.warpAffine(current_frame, M, (current_frame.shape[1], current_frame.shape[0]))
            aligned_frames[b, s] = torch.from_numpy(aligned_frame).permute(2, 0, 1)
        
        all_shifts.append(sequence_shifts)
    
    return aligned_frames, all_shifts

def adjust_coordinates(coords, shifts):
    """
    Adjust coordinates based on computed shifts
    """
    batch_size, seq_length = coords.shape[0], coords.shape[1]
    adjusted_coords = coords.clone()
    
    for b in range(batch_size):
        for s in range(1, seq_length):
            shift = shifts[b][s-1]
            adjusted_coords[b, s, 0] -= shift[0]  # adjust x coordinate
            adjusted_coords[b, s, 1] -= shift[1]  # adjust y coordinate
    
    return adjusted_coords

def compute_bbox_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: First bbox [x_min, y_min, x_max, y_max]
        box2: Second bbox [x, y, w, h] (ground truth format)
    Returns:
        float: IoU score
    """
    try:
        # Convert ground truth box from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        x, y, w, h = box2
        gt_box = [x, y, x + w, y + h]
        
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = gt_box
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Calculate areas
        intersection = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = w * h  # ground truth area
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0
        
        return intersection / union
    except Exception as e:
        print(f"Error computing bbox IoU: {e}")
        return 0

def check_mask(mask, threshold=0.7):
    """
    Check if mask is valid, reject the mask(set it to all 0s) if its too large in a scene, which
    means its very likely to be a backgroud
    
    Args:
        mask: Binary mask to check, shape (H, W)
        threshold: Threshold for valid mask
    """
    global Rejected_cnt
    if mask.sum() > threshold * mask.shape[0] * mask.shape[1]:
        Rejected_cnt += 1 
        return np.zeros_like(mask)
    else:
        return mask

def check_bbox(bbox, imgSize, threshold=0.7):
    """Reject the bbox if it is too large against the whole frame

    Args:
        bbox (List): List of coords like: [int(x_min), int(y_min), int(x_max), int(y_max)]
        imgSize (_type_): the img width * height
        threshold (float, optional): The size ratio for rejection. Defaults to 0.7.
    """
    if bbox is None:
        return False
    
    # convert bbox to (x_min, y_min, x_max, y_max) format
    x_min, y_min, x_max, y_max = bbox
    bboxSize=(x_max-x_min)*(y_max-y_min)
    global Rejected_cnt
    
    if bboxSize > imgSize*threshold:
        Rejected_cnt += 1
        return False
    return True

