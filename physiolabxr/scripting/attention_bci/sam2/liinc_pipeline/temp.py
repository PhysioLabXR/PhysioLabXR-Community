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
        avg_mask = compute_average_mask(batch_masks[b], threshold=avg_threshold)
        
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

def save_sequence_results(frames, coords, frame_paths, sequence_results, save_dir="results"):
    """
    Save visualizations with IoU-based mask selection results
    """
    os.makedirs(save_dir, exist_ok=True)
    batch_size = frames.shape[0]
    seq_length = frames.shape[1]
    
    for batch_idx in range(frames.shape[0]):
        print(f"\nProcessing Batch {batch_idx}...")
        
        batch_dir = os.path.join(save_dir, f"batch_{batch_idx}")
        os.makedirs(batch_dir, exist_ok=True)
        log_path = os.path.join(batch_dir, "results.txt")
        
        for item_idx in range(batch_size):
            for seq_idx in range(seq_length):
                frame = frames[item_idx, seq_idx].permute(1,2,0).cpu().numpy().astype(np.uint8)
                
                # Get masks and metadata
                original_mask = sequence_results[seq_idx]['original_masks'][item_idx]
                final_mask = sequence_results[seq_idx]['final_masks'][item_idx]
                mask_info = sequence_results[seq_idx]['mask_sources'][item_idx]
                coord = sequence_results[seq_idx]['coords'][item_idx]
                frame_path = frame_paths[item_idx][seq_idx]
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
                
                # Original mask
                ax1.imshow(frame)
                show_mask(original_mask, ax1, borders=True)
                ax1.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)
                ax1.set_title(f"Original Mask - Item {item_idx}, Sequence {seq_idx}\n"
                            f"Frame: {os.path.basename(frame_path)}\n"
                            f"Coord: ({coord[0][0]:.1f}, {coord[0][1]:.1f})")
                ax1.axis('off')
                
                # Selected mask (original or average)
                ax2.imshow(frame)
                show_mask(final_mask, ax2, borders=True)
                ax2.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)
                ax2.set_title(f"Selected Mask ({mask_info['source']}) - IoU: {mask_info['iou_score']:.3f}\n"
                            f"Frame: {os.path.basename(frame_path)}\n"
                            f"Coord: ({coord[0][0]:.1f}, {coord[0][1]:.1f})")
                ax2.axis('off')
                
                # Save figure
                save_path = os.path.join(batch_dir, f"sequence_{item_idx}_{seq_idx}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close()
                
                # Log information
                with open(log_path, "a") as f:
                    f.write(f"Item {item_idx}, Sequence {seq_idx}:\n")
                    f.write(f"- Frame path: {frame_path}\n")
                    f.write(f"- Fixation coordinate: ({coord[0][0]:.1f}, {coord[0][1]:.1f})\n")
                    f.write(f"- Mask source: {mask_info['source']}\n")
                    f.write(f"- IoU score: {mask_info['iou_score']:.3f}\n")
                    if seq_idx > 0:
                        f.write(f"- Shifts: {sequence_results[seq_idx]['shifts'][item_idx]}\n")
                    f.write("\n")
                
                print(f"Saved sequence {seq_idx} for item {item_idx} in batch {batch_idx}")

        print(f"Completed processing batch {batch_idx}")