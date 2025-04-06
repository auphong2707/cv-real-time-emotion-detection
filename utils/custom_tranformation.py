import cv2
import numpy as np
from PIL import Image

class HistogramEqualization:
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:  # Grayscale check
            img_eq = cv2.equalizeHist(img_np)
            return Image.fromarray(img_eq)
        return img  # Leave as-is if not grayscale

class CLAHEEqualization:
    def __init__(self, clip_limit=4.0, tile_grid_size=(4, 4)):  # Stronger than before
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 2:
            img_eq = self.clahe.apply(img_np)
            return Image.fromarray(img_eq)
        return img

class BrightenIfTooDark:
    def __init__(self, threshold=50, boost=30):
        self.threshold = threshold
        self.boost = boost

    def __call__(self, img):
        img_np = np.array(img)
        if img_np.mean() < self.threshold:
            img_np = np.clip(img_np + self.boost, 0, 255)
        return Image.fromarray(img_np.astype(np.uint8))

class GammaCorrection:
    def __init__(self, gamma=1.5):  # >1 = brighter, <1 = darker
        self.gamma = gamma

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = np.power(img_np, 1 / self.gamma)
        img_np = (img_np * 255).astype(np.uint8)
        return Image.fromarray(img_np)
