# How to convert the data from attention experiments to a valid paligemma dataset

## Pre-requisites
use conda to create a virtual env. activate it.

## First, build the sam-2 library
`cd ..`

in the sam2 folder, do ` pip install -e . `

and then in the liinc pipeline folder, do `pip3 install -r requirements.txt`  

## Second, download the sam-2 ckpt from:
Default: hiera_large
- [sam2.1_hiera_tiny.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt)
- [sam2.1_hiera_small.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt)
- [sam2.1_hiera_base_plus.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt)
- [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt)

put them in `sam2/checkpoints` folder

## configure `PhysioLabXR-Community/physiolabxr/scripting/attention_bci/sam2/liinc_pipeline/sam_clicks.py` for convertion

```
dataloader = get_target_sequence_dataloader(
        # Leave your data folder here
        root_dir=r"/home/ian/participant=1_session=2",
        # Your target_id from the experiment, this works with target response dtn: 2 only for the target_id items
        target_id="407.0",
        batch_size=32,
        
        ```
        the sequence length of frames 
        where the fixations are on one single 
        instace of an item
        ```
        sequence_length=3,

        transform=transform
    )

    # Configure your device
    device = init_device(device_name='cuda:4')
```


**All of the areas with annotations should be Configured** 

**Again, based on your specific need, adjust the thresholds in `def check_mask(mask, threshold=0.7)` and `def check_bbox(bbox, imgSize, threshold=0.7):`, now the logic is adjusted to not include the data into the jsonl dataset for paligemma if the binary mask or the bounding box of an item is over 0.7 of the size of the whole frame.**