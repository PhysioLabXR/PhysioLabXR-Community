#!/usr/bin/env python3
import json
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import re

def extract_target_id_from_path(path):
    """Extract target_id from file path"""
    match = re.search(r'target_id=(\d+(\.\d+)?)', path)
    if match:
        return float(match.group(1))
    return None

def find_files_for_target(base_dir, target_id):
    """Find all relevant files for a specific target_id"""
    target_files = []
    
    # Look for target directory
    target_pattern = f"target_id={target_id}"
    for item in os.listdir(base_dir):
        if target_pattern in item:
            target_dir = os.path.join(base_dir, item)
            # Look for block directories
            for block_dir in os.listdir(target_dir):
                if re.search(r'id=\d+\.0$', block_dir):
                    block_path = os.path.join(target_dir, block_dir)
                    # Find fixations.jsonl file
                    fixations_file = os.path.join(block_path, "fixations.jsonl")
                    if os.path.exists(fixations_file):
                        target_files.append(fixations_file)
    
    return target_files

def parse_paligemma_bbox(bbox_str):
    """Parse paligemma bbox string to [y1, x1, y2, x2]"""
    numbers = [int(bbox_str[i:i+4]) for i in [4, 12, 20, 28]]
    return numbers

def convert_to_xyxy(bbox):
    """Convert [y1, x1, y2, x2] to [x1, y1, x2, y2]"""
    return [bbox[1], bbox[0], bbox[3], bbox[2]]

def calculate_iou(bbox1, bbox2):
    """Calculate IOU between two bboxes in [x1, y1, x2, y2] format"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def visualize_bbox_comparison(image_path1, image_path2, bbox1, bbox2, iou, save_path):
    """Visualize two bounding boxes side by side with IOU score"""
    try:
        plt.close('all')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Load and display first image
        img1 = plt.imread(image_path1)
        ax1.imshow(img1)
        ax1.set_title('Image 1')
        
        # Draw first bbox
        x1, y1, x2, y2 = bbox1
        width = x2 - x1
        height = y2 - y1
        rect1 = plt.Rectangle((x1, y1), width, height, fill=False, color='red')
        ax1.add_patch(rect1)
        
        # Load and display second image
        img2 = plt.imread(image_path2)
        ax2.imshow(img2)
        ax2.set_title('Image 2')
        
        # Draw second bbox
        x1, y1, x2, y2 = bbox2
        width = x2 - x1
        height = y2 - y1
        rect2 = plt.Rectangle((x1, y1), width, height, fill=False, color='blue')
        ax2.add_patch(rect2)
        
        plt.suptitle(f'IOU: {iou:.4f}')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Error creating visualization: {e}")

def process_directories(base_dir1, base_dir2, target_id, output_dir):
    """Process files from two base directories for a specific target_id"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # Find relevant files
    files1 = find_files_for_target(base_dir1, target_id)
    files2 = find_files_for_target(base_dir2, target_id)
    
    print(f"Found {len(files1)} files in dir1 and {len(files2)} files in dir2 for target_id {target_id}")
    
    # Read and process files
    entries1 = {}
    entries2 = {}
    
    # Process first directory
    for file_path in files1:
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                frame_path = entry['image']
                frame_name = os.path.basename(frame_path)
                entries1[frame_name] = entry
    
    # Process second directory
    for file_path in files2:
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                frame_path = entry['image']
                frame_name = os.path.basename(frame_path)
                entries2[frame_name] = entry
    
    # Calculate IOUs for matching frames
    results = []
    ious = []
    
    print("\nCalculating IOUs...")
    matches = set(entries1.keys()) & set(entries2.keys())
    print(f"Found {len(matches)} matching frames")
    
    for frame_name in tqdm.tqdm(matches):
        entry1 = entries1[frame_name]
        entry2 = entries2[frame_name]
        
        # Parse bboxes
        bbox1 = parse_paligemma_bbox(entry1['suffix'].split()[0])
        bbox2 = parse_paligemma_bbox(entry2['suffix'].split()[0])
        
        # Convert to [x1, y1, x2, y2] format for IOU calculation
        bbox1_xyxy = convert_to_xyxy(bbox1)
        bbox2_xyxy = convert_to_xyxy(bbox2)
        
        # Calculate IOU
        iou = calculate_iou(bbox1_xyxy, bbox2_xyxy)
        ious.append(iou)
        
        # Create result entry
        result = {
            'frame_name': frame_name,
            'iou': iou,
            'bbox1': bbox1_xyxy,
            'bbox2': bbox2_xyxy,
            'image1_path': entry1['image'],
            'image2_path': entry2['image']
        }
        results.append(result)
        
        # Create visualization
        viz_path = os.path.join(output_dir, 'visualizations', 
                              f'frame_{frame_name}_comparison.png')
        visualize_bbox_comparison(
            entry1['image'], 
            entry2['image'],
            bbox1_xyxy, 
            bbox2_xyxy,
            iou,
            viz_path
        )
    
    # Calculate and save statistics
    print("\nCalculating statistics...")
    with open(os.path.join(output_dir, f'iou_statistics_target_{target_id}.txt'), 'w') as f:
        f.write(f"IOU Statistics for Target ID {target_id}:\n\n")
        f.write(f"Number of matches: {len(ious)}\n")
        if ious:
            f.write(f"Mean IOU: {np.mean(ious):.4f}\n")
            f.write(f"Median IOU: {np.median(ious):.4f}\n")
            f.write(f"Min IOU: {np.min(ious):.4f}\n")
            f.write(f"Max IOU: {np.max(ious):.4f}\n")
            f.write(f"Std IOU: {np.std(ious):.4f}\n")
    
    # Save detailed results
    with open(os.path.join(output_dir, f'iou_results_target_{target_id}.jsonl'), 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    # Plot IOU distribution
    if ious:
        plt.figure(figsize=(10, 6))
        plt.hist(ious, bins=20, color='blue', alpha=0.7)
        plt.xlabel('IOU')
        plt.ylabel('Count')
        plt.title(f'IOU Distribution for Target {target_id}')
        plt.savefig(os.path.join(output_dir, f'iou_distribution_target_{target_id}.png'))
        plt.close()

def main():
    # Directly pass the parameters here
    dir1 = 'path/to/first/directory'
    dir2 = 'path/to/second/directory'
    target_id = fillin # Example target_id
    output = 'iou_analysis_results'

    process_directories(dir1, dir2, target_id, output)

# In a script, you'd just call main() like this:
if __name__ == "__main__":
    main()
