import os
import json
import cv2
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path

@dataclass
class Fixation:
    """
    Data class representing a single eye tracking fixation.
    
    Attributes:
        item_name (str): The name of the item being viewed
        item_id (float): The ID signifying the type of the item
        dtn (float): 1: Distractor, 2: Target 3: Novelty
        block_id (float): The block ID for the experimentation block
        fixation_i_in_block (int): The index of the fixation in the block
        item_index (int): The collision box index of an item in the scene
        gaze_x (int): Left to right gaze position
        gaze_y (int): Top to bottom gaze position
        local_clock (float): Timestamp of the fixation
        frame_number (int): Frame number
        frame_filename (str): Frame file name
    """
    item_name: str
    item_id: float
    dtn: float
    block_id: float
    fixation_i_in_block: int
    item_index: int
    gaze_x: int
    gaze_y: int
    local_clock: float
    frame_number: int
    frame_filename: str

@dataclass
class ItemFixationSequence:
    """
    Data class representing a sequence of fixations on the same item.
    
    Attributes:
        item_name (str): Name of the item being viewed
        item_id (float): ID of the item
        dtn (float): Item type (2: Target)
        block_id (float): Block ID
        fixations (List[Fixation]): List of sequential fixations on this item
    """
    item_name: str
    item_id: float
    dtn: float
    block_id: float
    fixations: List[Fixation]

class TargetSequentialDataset(Dataset):
    """
    Dataset for target (dtn=2) eye tracking sequences.
    
    Args:
        root_dir (str): Root directory containing target and block subdirectories
        target_id (str): Specific target ID to load
        transform: Optional transforms to apply to images
        sequence_length (int): Fixed length for each sequence
    """
    def __init__(
        self,
        root_dir: str,
        target_id: str,
        transform=None,
        sequence_length: int = 2
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Store sequences of fixations on target items
        self.fixation_sequences = []
        
        # Load only specified target
        target_pattern = f"target_id={target_id}_*"
        target_dirs = list(self.root_dir.glob(target_pattern))
        
        if not target_dirs:
            raise ValueError(f"No data found for target_id={target_id}")
        
        # Load data from all blocks in this target
        for block_dir in target_dirs[0].glob("block_i=*"):
            fixations_path = block_dir / "fixations.jsonl"
            if not fixations_path.exists():
                raise FileNotFoundError(f"Fixations file not found: {fixations_path}")
            
            # Load and group fixations by item_id
            block_sequences = self._create_item_sequences(
                fixations_path=fixations_path,
                block_dir=block_dir
            )
            self.fixation_sequences.extend(block_sequences)

    def _load_fixations(self, fixations_path: Path) -> List[Fixation]:
        """Load and parse fixations from JSONL file."""
        fixations = []
        with open(fixations_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                # Only include target (dtn=2) fixations
                if data['dtn'] == 2.0:
                    fixation = Fixation(
                        item_name=data['item_name'],
                        item_id=data['item_id'],
                        dtn=data['dtn'],
                        block_id=data['block_id'],
                        fixation_i_in_block=data['fixation_i_in_block'],
                        item_index=data['item_index'],
                        gaze_x=data['GazePixelPositionX'],
                        gaze_y=data['GazePixelPositionY'],
                        local_clock=data['LocalClock'],
                        frame_number=data['FrameNumber'],
                        frame_filename=data['FrameFileName']
                    )
                    fixations.append(fixation)
        
        return sorted(fixations, key=lambda x: x.frame_number)

    def _create_item_sequences(
        self,
        fixations_path: Path,
        block_dir: Path
    ) -> List[Dict]:
        """Create sequences of target fixations based on item_index."""
        current_sequence = []
        sequences = []
        current_item_index = None
        
        fixations = self._load_fixations(fixations_path)
        
        for fixation in fixations:
            if current_item_index is None:
                current_item_index = fixation.item_index
                current_sequence = [fixation]
            elif fixation.item_index == current_item_index:
                current_sequence.append(fixation)

            # processing when met with a different item index
            else:
                # Process previous sequence
                if current_sequence:
                    # If sequence is longer than desired length, take evenly spaced frames
                    if len(current_sequence) >= self.sequence_length:
                        indices = np.linspace(0, len(current_sequence)-1, self.sequence_length, dtype=int)
                        selected_fixations = [current_sequence[i] for i in indices]
                    else:
                        # If sequence is shorter, duplicate last frame to reach desired length
                        selected_fixations = current_sequence + [current_sequence[-1]] * (self.sequence_length - len(current_sequence))
                    
                    sequences.append({
                        'sequence': ItemFixationSequence(
                            item_name=current_sequence[0].item_name,
                            item_id=current_sequence[0].item_id,
                            dtn=current_sequence[0].dtn,
                            block_id=current_sequence[0].block_id,
                            fixations=selected_fixations
                        ),
                        'block_dir': block_dir
                    })
                
                # Start new sequence
                current_item_index = fixation.item_index
                current_sequence = [fixation]
        
        # Don't forget the last sequence
        if current_sequence is not None:
            if len(current_sequence) >= self.sequence_length:
                indices = np.linspace(0, len(current_sequence)-1, self.sequence_length, dtype=int)
                selected_fixations = [current_sequence[i] for i in indices]
            else:
                selected_fixations = current_sequence + [current_sequence[-1]] * (self.sequence_length - len(current_sequence))
            
            sequences.append({
                'sequence': ItemFixationSequence(
                    item_name=current_sequence[0].item_name,
                    item_id=current_sequence[0].item_id,
                    dtn=current_sequence[0].dtn,
                    block_id=current_sequence[0].block_id,
                    fixations=selected_fixations
                ),
                'block_dir': block_dir
            })
        
        return sequences

    def __len__(self) -> int:
        """Get total number of fixation sequences."""
        return len(self.fixation_sequences)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sequence of fixations on the same target item_index instance."""
        sequence_info = self.fixation_sequences[idx]
        sequence = sequence_info['sequence']
        block_dir = sequence_info['block_dir']
        
        frames = []
        frame_paths = []
        
        for fixation in sequence.fixations:
            frame_path = block_dir / fixation.frame_filename
            frame_paths.append(str(frame_path))
            
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        item = {
            'sequence': sequence,
            'frames': np.stack(frames),
            'frame_paths': frame_paths,
            'block_dir': str(block_dir)
        }
        
        if self.transform:
            item = self.transform(item)
        
        return item

def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function that returns:
    - frames tensor of shape (B, S, 3(rgb), H, W)
    - tensor of shape (B, S, 2) containing fixation coordinates
    - list of frame paths for each sequence
    """
    batch_size = len(batch)
    if batch_size == 0:
        return (torch.tensor([]), torch.tensor([]), [])
    
    seq_len, channels, height, width = batch[0]['frames'].shape
    
    frames = torch.zeros(batch_size, seq_len, channels, height, width)
    fixation_coords = torch.zeros(batch_size, seq_len, 2)
    frame_paths = []  # Store paths for each sequence
    
    for i, item in enumerate(batch):
        frames[i,:] = item['frames']
        sequence = item['sequence']
        batch_paths = []  # Store paths for current sequence
        
        for j in range(len(sequence.fixations)):
            fixation_coords[i,j,0] = sequence.fixations[j].gaze_x
            fixation_coords[i,j,1] = sequence.fixations[j].gaze_y
            batch_paths.append(item['frame_paths'][j])
            
        frame_paths.append(batch_paths)
    
    return frames, fixation_coords, frame_paths
    
    
    
    

def get_target_sequence_dataloader(
    root_dir: str,
    target_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    sequence_length: int = 2,
    transform=None,
) -> DataLoader:
    """Create a DataLoader for target sequence data."""
    dataset = TargetSequentialDataset(
        root_dir=root_dir,
        target_id=target_id,
        transform=transform,
        sequence_length=sequence_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

class ToTensor:
    """Convert ndarrays to Tensors"""
    def __call__(self, item):
        if 'frames' in item:
            item['frames'] = torch.from_numpy(item['frames'].transpose(0, 3, 1, 2))#.float()
        return item


if __name__ == '__main__':
    transform = ToTensor()
    target_id = "104.0"
    
    try:
        dataloader = get_target_sequence_dataloader(
            root_dir=r"/home/ian/participant=1_session=2",
            target_id=target_id,
            batch_size=32,
            sequence_length=2,
            transform=transform
        )
        
        for frames, coords in dataloader:
            print(f"Frames shape: {frames.shape}")  # (B, S, C, H, W)
            
            print(f"Coordinates shape: {coords.shape}")  # (B, S, 2)
            print(f"Sample coordinates: X={coords[0,0,0]}, Y={coords[0,0,1]}")  # First image coords
            
            #do not break in actual pipeline, for demonstration purposes only
            # Check tensor values and shape
            print("Tensor shape:", frames[0,0,:,:,:].shape)
            print("Tensor min value:", frames[0,0,:,:,:].min())
            print("Tensor max value:", frames[0,0,:,:,:].max())
            import matplotlib.pyplot as plt
            print(frames[0,0,:,:,:])
            plt.figure(figsize=(8,8))
            plt.imshow(frames[0,0,:,:,:].permute(1,2,0).cpu().numpy().astype('uint8'))
            plt.axis('on')
            plt.savefig(f"sample_image_{target_id}.png")
            plt.close()
            break
            
    except Exception as e:
        print(f"Error: {e}")