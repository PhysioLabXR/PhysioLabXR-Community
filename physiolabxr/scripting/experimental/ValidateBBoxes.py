import os

import pandas as pd
import cv2
import json

from physiolabxr.examples.Eyetracking.EyeUtils import clip_bbox, add_bounding_box

data_root = r"C:\Users\LLINC-Lab\Documents\Recordings\ReNaUnityCameraCapture_10_31_2024_23_01_33"
image_shape = (448, 448, 3)

# read GazeInfo.csv
df = pd.read_csv(os.path.join(data_root, "GazeInfo.csv"))

# get the rows where bbox column is not '{}'
df = df[df['bboxes'] != '{}']

# show the camera capture image with the bounding boxes for each row
# for each row, load the image and draw the bounding boxes


for idx, row in df.iterrows():
    img = cv2.imread(os.path.join(data_root, f"{row['FrameNumber']}.png"))
    bboxes = json.loads(row['bboxes'])
    for item_index, item_bbox in bboxes.items():
        # clip the bbox to the image size, the bbox is in the format x_center, y_center, width, height
        item_bbox_clipped = clip_bbox(*item_bbox, image_shape)
        img = add_bounding_box(img, *item_bbox_clipped, color=(0, 255, 0))
        # put the item index as text
        cv2.putText(img, str(item_index), (item_bbox_clipped[0], item_bbox_clipped[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
