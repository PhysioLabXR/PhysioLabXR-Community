"""
This class is responsible for loading the data and run inference on the data with SAM-2
"""
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.gather_raw_data import ToTensor,get_target_sequence_dataloader

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

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

def compute_average_mask(masks):
    """
    Compute average mask across sequence
    """
    return np.mean(masks, axis=0)

def reverse_shift_mask(mask, shift):
    """
    Apply reverse shift to mask
    """
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(mask.astype(np.float32), M, (mask.shape[1], mask.shape[0]))

def process_sequence_batch(model_predictor, frames, coords):
    batch_size = frames.shape[0] 
    seq_length = frames.shape[1] 
    results = []

    # Align frames and adjust coordinates
    aligned_frames, shifts = align_sequence(frames)
    adjusted_coords = adjust_coordinates(coords, shifts)

    for seq_idx in range(seq_length):
        current_frames = aligned_frames[:,seq_idx]  
        current_coords = adjusted_coords[:,seq_idx]  
    
        image_list = []
        point_coords_batch = []
        point_labels_batch = []

        for b in range(batch_size):
            img = current_frames[b].permute(1,2,0).cpu().numpy().astype(np.uint8)
            image_list.append(img)
            
            coord = current_coords[b].cpu().numpy()
            point_coords_batch.append(np.array([coord]))  
            point_labels_batch.append(np.array([1]))
            
        model_predictor.set_image_batch(image_list)
        
        masks, iou_predictions, low_res_masks = model_predictor.predict_batch(
            point_coords_batch=point_coords_batch,
            point_labels_batch=point_labels_batch,
            multimask_output=True
        )
        
        # For each batch item, compute average mask across sequence
        for b in range(batch_size):
            best_masks = [mask[np.argmax(score)] for mask, score in zip(masks, iou_predictions)]
            avg_mask = compute_average_mask(best_masks)
            
            # Reverse shifts for individual masks
            original_masks = []
            for s in range(seq_length):
                if s == 0:
                    original_masks.append(best_masks[s])
                else:
                    shift = shifts[b][s-1]
                    original_masks.append(reverse_shift_mask(best_masks[s], shift))
        
        frame_results = {
            'masks': masks,
            'scores': iou_predictions,
            'low_res_masks': low_res_masks,
            'coords': point_coords_batch,
            'shifts': shifts,
            'average_masks': avg_mask,
            'original_masks': original_masks
        }
        results.append(frame_results)
        
    return results

def save_sequence_results(frames, coords, sequence_results, save_dir="results"):
    """
    Save visualizations for all frames and their corresponding masks.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for seq_idx in range(frames.shape[1]):
        print(f"\nProcessing Sequence {seq_idx}...")
        
        seq_dir = os.path.join(save_dir, f"sequence_{seq_idx}")
        os.makedirs(seq_dir, exist_ok=True)
        
        for batch_idx in range(frames.shape[0]):
            frame = frames[batch_idx,seq_idx].permute(1,2,0).cpu().numpy().astype(np.uint8)
            
            # Get original and average masks
            original_mask = sequence_results[seq_idx]['original_masks'][batch_idx]
            average_mask = sequence_results[seq_idx]['average_masks'][batch_idx]
            coord = sequence_results[seq_idx]['coords'][batch_idx]
            
            # Create visualization with both masks
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
            
            # Original mask
            ax1.imshow(frame)
            show_mask(original_mask, ax1, borders=True)
            ax1.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)
            ax1.set_title("Original Mask")
            ax1.axis('off')
            
            # Average mask
            ax2.imshow(frame)
            show_mask(average_mask, ax2, borders=True)
            ax2.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)
            ax2.set_title("Average Mask")
            ax2.axis('off')
            
            # Save figure
            save_path = os.path.join(seq_dir, f"batch_{batch_idx}.png")
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Save shifts to log file
            log_path = os.path.join(seq_dir, "results.txt")
            with open(log_path, "a") as f:
                f.write(f"Batch {batch_idx}:\n")
                f.write(f"- Shifts: {sequence_results[seq_idx]['shifts'][batch_idx]}\n")
                f.write(f"- Fixation coordinate: ({coord[0][0]:.1f}, {coord[0][1]:.1f})\n\n")
            
            print(f"Saved batch {batch_idx}")

[Rest of the file remains unchanged...]
