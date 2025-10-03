import zmq
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.bind("tcp://127.0.0.1:5556")


width, height = 1000, 1000
rgba_matrix = np.random.randint(0, 256, (height, width, 4), dtype=np.uint8)

time_start = time.time()
for i in range(6):
    cv2.imwrite('testimage.png', rgba_matrix)

time_end = time.time()
print(f'cv2.imwrite took {time_end - time_start} seconds')





# socket.send_multipart([array1.tobytes(), array2.tobytes(), array3.tobytes()])




