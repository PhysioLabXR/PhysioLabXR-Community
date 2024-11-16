"""
This class is responsible for loading the data and run inference on the data with SAM-2
"""
import json
import sys
import os
from typing import List, Dict
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.gather_raw_data import ToTensor,get_target_sequence_dataloader

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

def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union between two binary masks.
    
    Args:
        mask1, mask2: Binary masks to compare
    Returns:
        float: IoU score
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0
    return np.sum(intersection) / np.sum(union)

def compute_average_binary_mask(masks,threshold=0.5):
    """
    Compute average mask by adding masks together and normalizing.
    This identifies areas where masks consistently overlap across the sequence.
    
    Args:
        masks (list): List of binary masks from a sequence
        
    Returns:
        np.ndarray: Average mask where values represent the fraction of masks that 
                   include each pixel (0-1 range)
    """
    if not masks:
        raise ValueError("Empty mask list provided")
    
    # Initialize accumulated mask with zeros in the shape of first mask
    accumulated_mask = np.zeros_like(masks[0], dtype=np.float32)
    
    # Add all masks together
    for mask in masks:
        accumulated_mask += mask.astype(np.float16)
    
    # Normalize by number of masks to get fraction of overlap (0-1 range)
    average_mask = accumulated_mask / len(masks)
    
    binary_mask=(average_mask >= threshold)
    print(f"Individual mask shape: {masks[0].shape}")
    print(f"Accumulated mask range: [{accumulated_mask.min()}, {accumulated_mask.max()}]")

    return binary_mask

def reverse_shift_mask(mask, shift):
    """
    Apply reverse shift to mask
    """
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    result = cv2.warpAffine(mask.astype(np.float32), M, (mask.shape[1], mask.shape[0]))
    # print(f"Mask shape before shift: {mask.shape}")
    # print(f"Mask shape after shift: {result.shape}")
    return result

def init_device(device_name:str='cuda:4'):
    """
    Initialize the compute env,
    """
    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device(device_name)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device
    
def load_data():
    transform = ToTensor()
    
    # Create dataloader
    train_loader = get_target_sequence_dataloader(
        root_dir="/path/to/your/data",
        target_id="104.0",
        batch_size=32,
        sequence_length=2,
        transform=transform,
        num_workers=4
    )
    return train_loader

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True):
    """
    Display a mask overlay on an axis
    """
    # Convert tensor to numpy if needed
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=75):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def check_mask(mask, threshold=0.7):
    """
    Check if mask is valid, reject the mask(set it to all 0s) if its too large in a scene, which
    means its very likely to be a backgroud
    
    Args:
        mask: Binary mask to check, shape (H, W)
        threshold: Threshold for valid mask
    """
    global Rejected_cnt
    #print("MSKSHAPE:",mask.shape)
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
    
    x_min, y_min, x_max, y_max = bbox
    bboxSize=(x_max-x_min)*(y_max-y_min)
    global Rejected_cnt
    
    if bboxSize > imgSize*threshold:
        Rejected_cnt += 1
        return False
    return True
    
def process_sequence_batch(model_predictor, frames, coords, iou_threshold=0.7, avg_threshold=0.5):
    """
    Process batch with IoU-based mask selection.
    
    Args:
        model_predictor: SAM model predictor
        frames: Input frames tensor
        coords: Input coordinates tensor
        iou_threshold: Threshold for selecting original vs average mask
        avg_threshold: Threshold for creating binary average mask
    """
    batch_size = frames.shape[0] 
    seq_length = frames.shape[1] 
    results = []

    # Align frames and adjust coordinates
    aligned_frames, shifts = align_sequence(frames)
    adjusted_coords = adjust_coordinates(coords, shifts)

    # Initialize structures to store masks
    batch_masks = [[] for _ in range(batch_size)]
    batch_original_masks = [[] for _ in range(batch_size)]

    # First pass: get all masks and store them
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
        
        frame_results = {
            'masks': masks,
            'scores': iou_predictions,
            'low_res_masks': low_res_masks,
            'coords': point_coords_batch,
            'shifts': shifts,
            'original_masks': [],
            'final_masks': [],  # Will store the selected masks (original or average)
            'mask_sources': []  # Will store whether we used original or average mask
        }
        
        # Store masks for this sequence
        for b in range(batch_size):
            best_mask_idx = np.argmax(iou_predictions[b])
            best_mask = masks[b][best_mask_idx]
            
            #Reject the mask if it has weird shape/too large that its very likely to be a background
            best_mask=check_mask(best_mask,threshold=0.5)
            
            
            
            batch_masks[b].append(best_mask)
            
            # Store original (reverse-shifted) mask
            if seq_idx == 0:
                frame_results['original_masks'].append(best_mask)
                batch_original_masks[b].append(best_mask)
            else:
                shift = shifts[b][seq_idx-1]
                reversed_mask = reverse_shift_mask(best_mask, shift)
                frame_results['original_masks'].append(reversed_mask)
                batch_original_masks[b].append(reversed_mask)
        
        results.append(frame_results)
    
    # Second pass: compute average masks and make selection based on IoU
    for b in range(batch_size):
        # Compute binary average mask for this sequence
        avg_mask = compute_average_binary_mask(batch_masks[b], threshold=avg_threshold)
        
        # For each sequence, compare original vs average mask
        for seq_idx in range(seq_length):
            original_mask = results[seq_idx]['original_masks'][b]
            
            # Shift average mask to match original frame position
            if seq_idx == 0:
                shifted_avg_mask = avg_mask
            else:
                shift = shifts[b][seq_idx-1]
                shifted_avg_mask = reverse_shift_mask(avg_mask, shift)
            
            # Compute IoU between original and shifted average mask
            iou_score = compute_iou(original_mask, shifted_avg_mask)
            
            # Select mask based on IoU threshold
            if iou_score > iou_threshold:
                selected_mask = original_mask
                mask_source = 'original'
            else:
                selected_mask = shifted_avg_mask
                mask_source = 'average'
            
            # Store selected mask and its source
            results[seq_idx]['final_masks'].append(selected_mask)
            results[seq_idx]['mask_sources'].append({
                'source': mask_source,
                'iou_score': iou_score
            })

    return results


def compute_bbox_from_mask(mask):
    """
    Compute bounding box from a binary mask.
    
    Args:
        mask: Binary mask array
    Returns:
        list: [x_min, y_min, x_max, y_max] or None if no mask found
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the bounding box that encompasses all contours
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = float('-inf'), float('-inf')
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = int(min(x_min, x))
        y_min = int(min(y_min, y))
        x_max = int(max(x_max, x + w))
        y_max = int(max(y_max, y + h))
    
    return [int(x_min), int(y_min), int(x_max), int(y_max)]

def create_jsonl_entry(frame_path, bbox, img_height, img_width, categories="target"):
    """
    Create a JSONL entry in specified format.
    
    Args:
        frame_path: Path to the original frame
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max]
        img_height: original image height
        img_width: original image width
        categories: categories string (semicolon separated)
    Returns:
        dict: JSONL entry
    """
    
    normalized_bbox = normalize_bbox(bbox,img_height,img_width)
    loc_tags=f"<loc{normalized_bbox[0]:04d}><loc{normalized_bbox[1]:04d}><loc<loc{normalized_bbox[2]:04d}><loc<loc{normalized_bbox[3]:04d}> {categories}"

    jsonl_entry = {"image": frame_path, "prefix": f"detect {categories}", "suffix": loc_tags}
    
    return jsonl_entry
    
    
    
def normalize_bbox(bbox, img_height, img_width, target_size=1024):
    """
    Normalize bounding box coordinates to 1024x1024 scale.
    Convert from [x_min, y_min, x_max, y_max] to [y1, x1, y2, x2]
    
    Args:
        bbox: [x_min, y_min, x_max, y_max] in original image coordinates
        img_height: original image height
        img_width: original image width
        target_size: target size for normalization (default 1024)
    Returns:
        list: [y1, x1, y2, x2] normalized to target_size, this is the paligemma jsonl representation
    """
    
    #explode bbox
    x_min, y_min, x_max, y_max = bbox
    
    # Normalize coordinates
    x1 = int((x_min / img_width) * target_size)
    y1 = int((y_min / img_height) * target_size)
    x2 = int((x_max / img_width) * target_size)
    y2 = int((y_max / img_height) * target_size)
    
    return [y1, x1, y2, x2]

def save_sequence_results(frames, coords, frame_paths, sequence_results, save_dir="results", categories="target", vis_interval=10):
    """
    Save visualizations and JSONL annotations for mask results
    Args:
        frames: input frames
        coords: input coordinates
        frame_paths: paths to frames
        sequence_results: detection results
        save_dir: directory to save results
        categories: detection categories
        vis_interval: save visualization every n frames (default: 10)
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = frames.shape[0]
    seq_length = frames.shape[1]
    
    # Create JSONL file for annotations
    jsonl_path = os.path.join(save_dir, "annotations.jsonl")
    
    # Create a single directory for visualizations
    vis_dir = os.path.join(save_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a single log file
    log_path = os.path.join(save_dir, "results.txt")
    
    total_frames = 0  # Counter for all processed frames
    
    for batch_idx in range(frames.shape[0]):
        print(f"\nProcessing Batch {batch_idx}...")
        
        for item_idx in range(batch_size):
            for seq_idx in range(seq_length):
                frame = frames[item_idx, seq_idx].permute(1,2,0).cpu().numpy().astype(np.uint8)
                img_height, img_width = frame.shape[0], frame.shape[1]
                img_size=img_height*img_width
                
                # Get masks and metadata
                final_mask = sequence_results[seq_idx]['final_masks'][item_idx]
                mask_info = sequence_results[seq_idx]['mask_sources'][item_idx]
                coord = sequence_results[seq_idx]['coords'][item_idx]
                frame_path = frame_paths[item_idx][seq_idx]
                
                # Compute bounding box from final mask
                bbox = compute_bbox_from_mask(final_mask)
                bbox is None if not check_bbox(bbox,imgSize=img_size,threshold=0.7) else bbox
                
                if bbox is not None:
                    # Create and save JSONL entry
                    jsonl_entry = create_jsonl_entry(
                        frame_path=frame_path,
                        bbox=bbox,
                        img_height=img_height,
                        img_width=img_width,
                        categories=categories
                    )
                    
                    with open(jsonl_path, 'a') as f:
                        json.dump(jsonl_entry, f)
                        f.write('\n')
                
                # Save visualization only for every vis_interval frames
                if total_frames % vis_interval == 0:
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    
                    # Show frame with mask and bounding box
                    ax.imshow(frame)
                    show_mask(final_mask, ax, borders=True)
                    if bbox is not None:
                        show_box(bbox, ax)
                    ax.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)
                    ax.set_title(f"Frame: {os.path.basename(frame_path)}\n"
                               f"Mask Source: {mask_info['source']}\n"
                               f"IoU: {mask_info['iou_score']:.3f}")
                    ax.axis('off')
                    
                    # Save figure with batch and sequence info
                    save_path = os.path.join(vis_dir, f"frame_{total_frames}_batch{batch_idx}_seq{seq_idx}.png")
                    plt.savefig(save_path, bbox_inches='tight', dpi=150)
                    plt.close()
                
                # Log information
                with open(log_path, "a") as f:
                    f.write(f"Frame {total_frames} (Batch {batch_idx}, Item {item_idx}, Seq {seq_idx}):\n")
                    f.write(f"- Frame path: {frame_path}\n")
                    f.write(f"- Fixation coordinate: ({coord[0][0]:.1f}, {coord[0][1]:.1f})\n")
                    f.write(f"- Mask source: {mask_info['source']}\n")
                    f.write(f"- IoU score: {mask_info['iou_score']:.3f}\n")
                    if bbox is not None:
                        norm_bbox = normalize_bbox(bbox, img_height, img_width)
                        f.write(f"- Original bbox: {bbox}\n")
                        f.write(f"- Normalized bbox (y1,x1,y2,x2): {norm_bbox}\n")
                    if seq_idx > 0:
                        f.write(f"- Shifts: {sequence_results[seq_idx]['shifts'][item_idx]}\n")
                    f.write("\n")
                
                total_frames += 1
                
        print(f"Completed processing batch {batch_idx}")
    
    print(f"Total frames processed: {total_frames}")
    print(f"Visualizations saved: {total_frames // vis_interval + 1}")

def init_sam_2(device):
    """init SAM-2 model
    Args:
        device (): the compute device for the model
    Returns:
        the sam model predictor: 
    """
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print(os.getcwd())
    sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor

if __name__ == "__main__":
    # Example usage with updated code
    transform = ToTensor()
    dataloader = get_target_sequence_dataloader(
        # Leave your data root folder here
        root_dir=r"/home/ian/participant=1_session=2",
        # Your target_id from the experiment, this works with target response dtn: 2 only for the target_id items
        target_id="607.0",
        batch_size=32,
        sequence_length=3,
        transform=transform
    )

    device = init_device(device_name='cuda:4')
    model_predictor = init_sam_2(device)

    # Main execution
    for frames, coords, frame_paths in dataloader:
        print("Processing batch...")
        
        # Process the batch
        sequence_results = process_sequence_batch(model_predictor, frames, coords)
        
        # Save results with path information
        categories = "target"  # currently only support single category
        save_sequence_results(frames, coords, frame_paths, sequence_results, 
                         save_dir="sam2_results", 
                         categories=categories)
        print("\nResults saved to 'sam2_results' directory")
    print("-"*32)
    print("All batches processed, the number of rejected masks: ", Rejected_cnt)
    print("-"*32)