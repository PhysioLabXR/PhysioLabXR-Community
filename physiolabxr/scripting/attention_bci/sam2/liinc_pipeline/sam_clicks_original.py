"""
This class is responsible for loading the data and run inference on the data with SAM-2
"""
#

# if using Apple MPS, fall back to CPU for unsupported ops

import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.gather_raw_data import ToTensor,get_target_sequence_dataloader

import numpy as np
import torch

import matplotlib.pyplot as plt


from PIL import Image

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
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
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
    Args:
        mask: numpy array or tensor mask
        ax: matplotlib axis
        random_color: whether to use random color for mask
        borders: whether to show mask borders
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
        import cv2
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

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.plot()
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        

def example_img_inference(model_predictor=None,image_path='/home/ian/participant=1_session=2/target_id=407.0_name=monitor/block_i=0_id=3.0/FixationIndex=8_FrameNumber=10241.0.png'):
    device = init_device()
    if model_predictor is None:
        model_predictor = init_sam_2(device)
    #image = Image.open(r'/home/ian/participant=1_session=2/target_id=407.0_name=monitor/block_i=0_id=3.0/FixationIndex=8_FrameNumber=10241.0.png')
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    plt.title("Original Image", fontsize=18)
    plt.savefig("original_image.png")
    
    image_tensor=torch.from_numpy(image).permute(2, 0, 1).float().to(device)
    model_predictor.set_image(image_tensor)
    X_coord,Y_coord=247,184
    input_point = np.array([[X_coord, Y_coord]])
    
    input_label = np.array([1])
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()
    plt.savefig("clicked_image.png")  
    print(model_predictor._features["image_embed"].shape, model_predictor._features["image_embed"][-1].shape)
    
    print(input_point)
    print(input_label)
    
    
    masks, scores, logits = model_predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    #masks.shape  # (number_of_masks) x H x W
    #take the first mask since its of the highest score
    mask=masks[0]
    score=scores[0]
    
    point_coords=input_point
    borders=False
    input_labels = input_label
    show_mask(mask, plt.gca(), borders=borders)
    plt.title(f"Mask {0}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
    plt.savefig("segmentated_mask(the highest_prob).png")  
        
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

def process_sequence_batch(model_predictor, frames, coords):
    batch_size = frames.shape[0] 
    seq_length = frames.shape[1] 
    results = []

    for seq_idx in range(seq_length):
        current_frames = frames[:,seq_idx]  
        current_coords = coords[:,seq_idx]  
    

        
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
            'coords': point_coords_batch
        }
        results.append(frame_results)
        
    return results

def save_sequence_results(frames, coords, sequence_results, save_dir="results"):
   """
   Save visualizations for all frames and their corresponding masks.
   
   Args:
       frames: tensor of shape [B,S,C,H,W] containing image frames
       coords: tensor of shape [B,S,2] containing fixation coordinates
       sequence_results: list of dictionaries containing mask prediction results
       save_dir: directory to save visualization images
   """
   # Create save directory if it doesn't exist
   os.makedirs(save_dir, exist_ok=True)
   
   for seq_idx in range(frames.shape[1]):  # iterate through sequences
       print(f"\nProcessing Sequence {seq_idx}...")
       
       # Create sequence subdirectory
       seq_dir = os.path.join(save_dir, f"sequence_{seq_idx}")
       os.makedirs(seq_dir, exist_ok=True)
       
       for batch_idx in range(frames.shape[0]):  # iterate through batch items
           # Get the frame image
           frame = frames[batch_idx,seq_idx].permute(1,2,0).cpu().numpy().astype(np.uint8)
           
           # Get prediction results
           masks = sequence_results[seq_idx]['masks'][batch_idx]  
           scores = sequence_results[seq_idx]['scores'][batch_idx]
           coord = sequence_results[seq_idx]['coords'][batch_idx]
           
           # Get highest scoring mask
           best_mask_idx = np.argmax(scores)
           best_mask = masks[best_mask_idx]
           best_score = scores[best_mask_idx]
           
           # Create visualization
           plt.figure(figsize=(10,10))
           plt.imshow(frame)  # Display original image
           show_mask(best_mask, plt.gca(), borders=True)  # Overlay mask
           plt.scatter(coord[0][0], coord[0][1], color='red', marker='*', s=200)  # Show fixation point
           
           # Add title with frame info
           plt.title(f"Sequence {seq_idx}, Batch {batch_idx}\nBest Score: {best_score:.3f}\nCoord: ({coord[0][0]:.1f}, {coord[0][1]:.1f})")
           plt.axis('off')
           
           # Save figure
           save_path = os.path.join(seq_dir, f"batch_{batch_idx}.png")
           plt.savefig(save_path, bbox_inches='tight', dpi=150)
           plt.close()  # Close figure to free memory
           
           # Save results to log file
           log_path = os.path.join(seq_dir, "results.txt")
           with open(log_path, "a") as f:
               f.write(f"Batch {batch_idx}:\n")
               f.write(f"- Score: {best_score:.3f}\n")
               f.write(f"- Fixation coordinate: ({coord[0][0]:.1f}, {coord[0][1]:.1f})\n\n")
           
           print(f"Saved batch {batch_idx} (Score: {best_score:.3f})")
           
if __name__ == "__main__":
    #example usage
    transform = ToTensor()
    dataloader = get_target_sequence_dataloader(
        root_dir=r"/home/ian/participant=1_session=2",
        target_id="407.0",
        batch_size=32,
        sequence_length=3,
        transform=transform
    )

    device = init_device()
    model_predictor = init_sam_2(device)



    # Main execution
    for frames, coords in dataloader:
        print("Processing batch...")
        
        # Process the batch
        sequence_results = process_sequence_batch(model_predictor, frames, coords)
            # [...,{
            #         'masks': masks,
            #         'scores': iou_predictions,
            #         'low_res_masks': low_res_masks,
            #         'coords': point_coords_batch
            # },]
        # Save all results
        save_sequence_results(frames, coords, sequence_results, save_dir="sam2_results")
        print("\nResults saved to 'sam2_results' directory")
    