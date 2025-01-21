import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import imageio
from albumentations import HorizontalFlip, VerticalFlip, GridDistortion, OpticalDistortion, CoarseDropout
from sklearn.utils import shuffle

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clahe_3d (img,gs): 
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a 3D image.

    This function converts an RGB image to the Lab color space, applies CLAHE 
    to the L channel to enhance contrast, and then converts the image back to RGB.

    Parameters:
        img (numpy.ndarray): Input RGB image.
        gs (int): Grid size for the CLAHE algorithm.

    Returns:
        numpy.ndarray: The contrast-enhanced RGB image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab) 
    clahe = cv2.createCLAHE(clipLimit=8,tileGridSize=(gs,gs)) 
    img[:,:,0] = clahe.apply(img[:,:,0]) 
    img = cv2.cvtColor(img, cv2.COLOR_Lab2RGB) 
    return img 

def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def augment_data(image, mask):
    """
    Applies the following data augmentations to the given image and mask:

    1. Horizontal flip
    2. Vertical flip
    3. Grid distortion
    4. Optical distortion

    Parameters:
        image (numpy.ndarray): Input RGB image.
        mask (numpy.ndarray): Input mask image.

    Returns:
        A tuple of two lists. The first list contains the augmented images, and
        the second list contains the corresponding augmented masks.
    """
    aug = HorizontalFlip(p=1.0)
    augmented = aug(image=image, mask=mask)
    x1 = augmented["image"]
    y1 = augmented["mask"]

    aug = VerticalFlip(p=1.0)
    augmented = aug(image=image, mask=mask)
    x2 = augmented["image"]
    y2 = augmented["mask"]


    aug = GridDistortion(p=1)
    augmented = aug(image=image, mask=mask)
    x4 = augmented['image']
    y4 = augmented['mask']

    aug = OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
    augmented = aug(image=image, mask=mask)
    x5 = augmented['image']
    y5 = augmented['mask']

    X = [x1, x2, x4, x5]
    Y = [y1, y2, y4, y5]

    return X, Y

def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    y = sorted(glob(os.path.join(path, "mask", "*.jpg")))
    return x, y


