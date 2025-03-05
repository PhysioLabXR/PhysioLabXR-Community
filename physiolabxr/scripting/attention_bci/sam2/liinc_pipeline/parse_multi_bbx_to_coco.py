from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from PIL import Image
import numpy as np
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

def center_to_corner(bbox):
   """Convert [x_center, y_center, width, height] to [x_left, y_top, width, height]"""
   x_c, y_c, w, h = bbox
   return [x_c - w/2, y_c - h/2, w, h]

def process_single_frame(args):
    """Process a single frame for multiprocessing"""
    frame_file, fixations, indices, target_id, block_id, block_path, output_dir = args
    
    frame_path = os.path.join(block_path, frame_file)
    frame_bboxes = []
    
    if not os.path.exists(frame_path):
        return None
    
    try:
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        img = plt.imread(frame_path)
        ax.imshow(img)
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)  # Reverse Y axis for display
        
        target_bboxes_found = False
        
        for fixation in fixations:
            bboxes = fixation['bboxes']
            
            for item_index in indices:
                str_idx = str(item_index)
                if str_idx in bboxes:
                    # Get original bbox (center format)
                    bbox = [float(x) for x in bboxes[str_idx]]
                    x_c, y_c, w, h = bbox
                    
                    # Convert to corner format for Rectangle
                    x_corner = x_c - w/2
                    y_corner = y_c - h/2
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (x_corner, y_corner), 
                        w, h, 
                        linewidth=2, 
                        edgecolor='r', 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add center point
                    ax.plot(x_c, y_c, 'rx', markersize=5)
                    
                    # Add text label near center
                    ax.text(x_c, y_c-5, f'Target {item_index}', 
                          color='red', fontsize=8, ha='center')
                    
                    target_bboxes_found = True
                    
                    # Store bbox info with original center format and full path
                    frame_bboxes.append({
                        'frame_file': frame_path,  # Store full path
                        'bbox_center': bbox,  # Keep original center format
                        'gaze': [float(fixation['GazePixelPositionX']), 
                                float(fixation['GazePixelPositionY'])],
                        'item_index': str_idx,
                        'target_id': float(target_id),
                        'block_id': int(block_id)
                    })

        if target_bboxes_found:
            for fixation in fixations:
                ax.plot(float(fixation['GazePixelPositionX']), 
                        float(fixation['GazePixelPositionY']), 
                        'bo', 
                        markersize=10,
                        label='Gaze Position')
            
            plt.title(f"Target {target_id}, Block {block_id}\n"
                    f"Frame: {frame_file}")
            
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            viz_path = os.path.join(output_dir, f"bbox_viz_{frame_file.split('.')[0]}.png")
            plt.savefig(viz_path, bbox_inches='tight', dpi=300)
   
    except Exception as e:
        print(f"Error processing frame {frame_file}: {e}")
        return None
    
    finally:
        plt.close('all')
    
    return frame_bboxes

def save_all_boxes_supervision(all_bboxes, output_path, id_name_catalog):
    """
    Save all bounding boxes in standard supervision format with corner coordinates
    and full image paths
    """
    dataset = {
        'images': [],
        'categories': [],
        'annotations': []
    }
    
    # Add categories
    for target_id, name in id_name_catalog.items():
        dataset['categories'].append({
            'id': int(target_id),
            'name': name
        })
    
    # Process all bboxes
    image_ids = set()
    annotation_id = 0
    
    # Group bboxes by frame
    frame_bboxes = defaultdict(list)
    for bbox_info in all_bboxes:
        frame_bboxes[bbox_info['frame_file']].append(bbox_info)
    
    # Process frame by frame
    for frame_file, bboxes in frame_bboxes.items():
        image_id = frame_file.split('.')[0]
        
        # Get full path from first bbox info (they're all from same directory)
        block_path = os.path.dirname(os.path.join(
            os.path.dirname(bboxes[0]['frame_file']), 
            frame_file
        ))
        full_image_path = os.path.join(block_path, frame_file)
        
        if image_id not in image_ids:
            dataset['images'].append({
                'id': image_id,
                'file_name': frame_file,
                'path': full_image_path,  # Include full path
                'width': 800,
                'height': 600
            })
            image_ids.add(image_id)
        
        for bbox_info in bboxes:
            # Convert center format to corner format for supervision
            corner_bbox = center_to_corner(bbox_info['bbox_center'])
            
            dataset['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': int(bbox_info['target_id']),
                'bbox': corner_bbox,  # [x_left, y_top, width, height]
                'confidence': 1.0
            })
            annotation_id += 1
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)


def process_fixations_and_visualize(target_indices, base_dir, id_name_catalog):
   """Process fixations.jsonl files and visualize bounding boxes"""
   all_bboxes = []
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

           output_dir = os.path.join(block_path, "bbox_visualizations")
           os.makedirs(output_dir, exist_ok=True)

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
                   block_id, block_path, output_dir
               ))

   # Process frames in parallel
   with mp.Pool(processes=mp.cpu_count()//4) as pool:
       results = list(tqdm.tqdm(
           pool.imap(process_single_frame, all_frame_args),
           total=len(all_frame_args),
           desc="Processing frames"
       ))
   
   # Collect all results
   for result in results:
       if result:
           all_bboxes.extend(result)

   # Save in supervision format
   all_bboxes_path = os.path.join(base_dir, "all_bboxes_supervision.json")
   save_all_boxes_supervision(all_bboxes, all_bboxes_path, id_name_catalog)

def main():
   base_dir = r"/home/ian/attentionbci-pilot_participant=2_session=0"
   logs_file = os.path.join(base_dir, "logs")
   catalog_file = os.path.join(base_dir, "Catalog")

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

   # Process fixations and create visualizations
   process_fixations_and_visualize(target_indices, base_dir, id_name_catalog)

if __name__ == "__main__":
   plt.switch_backend('agg')
   main()