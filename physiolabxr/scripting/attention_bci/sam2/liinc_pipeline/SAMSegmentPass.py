import numpy as np
import torch
import gc
from physiolabxr.scripting.RenaScript import RenaScript
from physiolabxr.scripting.attention_bci.sam2.liinc_pipeline import sam_clicks

class SAMSegmentPass(RenaScript):
    def __init__(self, *args, **kwargs):
        """
        Please do not edit this function
        """
        super().__init__(*args, **kwargs)

    # Start will be called once when the run button is hit.
    def init(self, rawDataPath: str,target_id: str, device_name: str, batch_size: int, sequence_length: int, save_dir: str, categories: str = "target"):
        """
        Summary:
        This function will be called to invoke the SAM2 (SAM2.1_hiera_large.pt ckpt)
        segmentation on fixation points when EEG has target responses.
        SAM-2 will run upon these points in selected frames to extract a segmentation mask,
        which is then used to convert to a bounding box before eventually exported as
        annotations for a Paligemma Jsonl Object Detectin Dataset format
                
        Args:
            rawDataPath (str): Path to the raw data folder containing fixations and frames.
                e.g.: "/home/ian/participant=1_session=2"
            target_id (str): The ID of the target category.
                Note this should align to one sub-folder of root folder
                For example:
                    participant=1_session=2/                <-- This is root folder
                        ├── target_id=1003.0_name=Crock2    <-- Specify your target_id here (1003.0)
                        │   ├── block_i=0_id=1.0                Then the program will iterate through all 
                        │   ├── block_i=0_id=66.0               sub-blocks in this subfolder here.
                        ├── target_id=104.0_name=Hybrid_1A
                        │   ├── block_i=0_id=6.0
                
            device_name (str): Your Comuting device. For example: 'cuda:2'
            
            batch_size (int): The size of the batches for SAM2. Default is 32.

            sequence_length (int): The length of the sequence for SAM2 data calibration.
                Note this can only be set to be above 1 in VR controlled env. This is 
                because this in VR env, we can track an object, and use these tracked frames 
                as a sequence, amongst which segementation masks were drawn and calibrated to
                be more accurate.
            
            save_dir (str): The directory to save the jsonl annotation(dataset for Paligemma)
            
            categories (str): The categories to include in the SAM2 output.
            Currently only support one category, being:"target"
        
        Returns: void
        
        """
        
        
        sam_clicks.process_target_sequence(
        root_dir=r"/home/ian/participant=1_session=2",
        target_id="607.0",
        device_name='cuda:2',
        batch_size=32,
        sequence_length=3,
        save_dir="sam2_results",# relative path to current dir
        categories="target"
    )
        print("data saved to " + save_dir)

    # loop is called <Run Frequency> times per second
    def loop(self):
        pass
        #print('Loop function is called')

    # cleanup is called when the stop button is hit
    def cleanup(self):
        # if torch.cuda.is_available():
        #     # Synchronize CUDA devices
        #     torch.cuda.synchronize()
            
        #     # Clear CUDA memory cache if requested
        #     torch.cuda.empty_cache()
                
        #     # Reset peak memory stats
        #     torch.cuda.reset_peak_memory_stats()
            
        #     # Clear CUDA memory allocator
        #     torch.cuda.empty_cache()
        
        #     # Run Python garbage collection
        #     gc.collect()
        #     print('Cleanup function is called')
        pass