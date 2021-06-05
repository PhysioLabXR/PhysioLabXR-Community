import numpy as np
import cv2
size = 720*16//9, 720
duration = 2
fps = 25
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
for _ in range(fps * duration):
    data = np.random.randint(0, 256, size, dtype='uint8')
    out.write(data)
out.release()