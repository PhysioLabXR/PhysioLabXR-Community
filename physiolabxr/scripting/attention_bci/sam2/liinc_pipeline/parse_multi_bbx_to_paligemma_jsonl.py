#!/usr/bin/env python3
from collections import defaultdict
import json
import os
import re
from PIL import Image
import multiprocessing as mp
from functools import partial
import tqdm

def load_id_name_catalog(catalog_file):
    """Read and parse catalog from file"""
    with open(catalog_file, 'r') as f:
        catalog_data = json.load(f)
    return {float(value): str(key) for key, value in catalog_data.items()}

def find_target_indices(logs_file):
    """Find item indices with dtn=2 in each block from logs file"""
    target_blocks = defaultdict(lambda: defaultdict(list))
    
    with open(logs_file, 'r') as f:
        blocks_data = json.load(f)
    
    for block_id, block_info in blocks_data.items():
        target_id = block_info['targets'][0]
        dtn_dict = block_info['ItemIndexCatalogDTNDict']
        target_indices = [
            int(idx) for idx, (item_id, dtn) in dtn_dict.items() 
            if dtn == 2
        ]
        target_blocks[target_id][int(block_id)] = target_indices
    
    return target_blocks

def find_matching_directory(target_id, block_id, base_dir):
    """Find matching directory using regex"""
    for dir_name in os.listdir(base_dir):
        if re.search(f"target_id={target_id}", dir_name):
            target_path = os.path.join(base_dir, dir_name)
            
            for block_dir in os.listdir(target_path):
                if re.search(f"id={block_id}.0$", block_dir):
                    return os.path.join(target_path, block_dir)
    return None

def convert_bbox_to_paligemma(bbox, img_width, img_height):
    """Convert [x_center, y_center, width, height] to [Y1, X1, Y2, X2] normalized to 1024x1024"""
    x_c, y_c, w, h = bbox
    
    # Convert to corner coordinates in original image space
    x1 = x_c - w/2
    y1 = y_c - h/2
    x2 = x1 + w
    y2 = y1 + h
    
    # Normalize to 1024x1024 space
    x1_norm = int(x1 * 1024 / img_width)
    y1_norm = int(y1 * 1024 / img_height)
    x2_norm = int(x2 * 1024 / img_width)
    y2_norm = int(y2 * 1024 / img_height)
    
    # Format as 4-digit coordinates
    return f"<loc{y1_norm:04d}><loc{x1_norm:04d}><loc{y2_norm:04d}><loc{x2_norm:04d}>"

def process_single_frame(args):
    """Process a single frame for JSONL output"""
    frame_file, fixations, indices, _, block_id, block_path = args
    
    frame_path = os.path.join(block_path, frame_file)
    frame_entries = []
    
    if not os.path.exists(frame_path):
        return None
    
    try:
        # Get image dimensions
        with Image.open(frame_path) as img:
            img_width, img_height = img.size
        
        found_targets = []
        
        for fixation in fixations:
            bboxes = fixation['bboxes']
            
            for item_index in indices:
                str_idx = str(item_index)
                if str_idx in bboxes:
                    bbox = [float(x) for x in bboxes[str_idx]]
                    paligemma_bbox = convert_bbox_to_paligemma(
                        bbox, img_width, img_height
                    )
                    found_targets.append((paligemma_bbox, "target"))
        
        if found_targets:
            # Create JSONL entry with full path
            entry = {
                'image': os.path.abspath(frame_path),  # 使用完整路径
                'prefix': 'detect target ; target ; target ; target ; target',
                'suffix': '; '.join([f"{bbox} {label}" for bbox, label in found_targets])
            }
            frame_entries.append(entry)
            
    except Exception as e:
        print(f"Error processing frame {frame_file}: {e}")
        return None
    
    return frame_entries

def process_fixations_and_save_jsonl(target_indices, base_dir, output_jsonl):
    """Process fixations and save in JSONL format"""
    all_frame_args = []
    
    # Collect all frame processing arguments
    for target_id, blocks in target_indices.items():
        for block_id, indices in blocks.items():
            block_path = find_matching_directory(target_id, block_id, base_dir)
            
            if block_path is None:
                print(f"No matching directory found for target {target_id}, block {block_id}")
                continue
            
            fixations_path = os.path.join(block_path, "fixations.jsonl")
            if not os.path.exists(fixations_path):
                print(f"No fixations file found in {block_path}")
                continue

            # Group fixations by frame
            frame_fixations = defaultdict(list)
            with open(fixations_path, 'r') as f:
                for line in f:
                    fixation = json.loads(line)
                    frame_file = fixation['FrameFileName']
                    frame_fixations[frame_file].append(fixation)
            
            # Create processing arguments
            for frame_file, fixations in frame_fixations.items():
                all_frame_args.append((
                    frame_file, fixations, indices, target_id, 
                    block_id, block_path
                ))

    # Process frames in parallel
    with mp.Pool(processes=mp.cpu_count()//4) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_single_frame, all_frame_args),
            total=len(all_frame_args),
            desc="Processing frames"
        ))
    
    # Save results in JSONL format
    with open(output_jsonl, 'w') as f:
        for result in results:
            if result:
                for entry in result:
                    f.write(json.dumps(entry) + '\n')

def main():
    base_dir = r"/home/ian/attentionbci-pilot_participant=2_session=0"
    logs_file = os.path.join(base_dir, "logs")
    catalog_file = os.path.join(base_dir, "Catalog")
    output_jsonl = os.path.join(base_dir, "paligemma_format.jsonl")

    # Load catalog
    id_name_catalog = load_id_name_catalog(catalog_file)
    print("First 10 items in catalog:")
    for id, name in list(id_name_catalog.items())[:10]:
        print(f"{id}: {name}")

    # Get target indices
    target_indices = find_target_indices(logs_file)
    print("\nTarget indices found:")
    for target_id, blocks in target_indices.items():
        print(f"\nTarget ID: {target_id}")
        for block_id, indices in blocks.items():
            print(f"  Block {block_id}: Target indices = {indices}")

    # Process fixations and save in JSONL format
    process_fixations_and_save_jsonl(target_indices, base_dir, output_jsonl)

if __name__ == "__main__":
    main()