import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt

from physiolabxr.scripting.AOIAugmentationScript.preprocessing.read_pickle import ImageInfo








read_pickle_file = 'data_dict.pkl'

with open(read_pickle_file, 'rb') as f:
    data_dict = pickle.load(f)

# test edge detection
