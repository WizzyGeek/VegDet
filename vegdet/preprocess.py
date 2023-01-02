
import cv2
import numpy as np

def preprocess(img: np.ndarray, width, height):
    img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (width, height))
