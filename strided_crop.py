import cv2
import numpy as np
import os
import tqdm
from data import augment_data

directories = [f'Crops',f'Crops/train/image',f'Crops/mask']
augment_list = ["horiz_", "vert_", "griddist_", "optdist_"]


def strided_crop(img, label, height, width, name, stride, directories):

    """
    Performs strided cropping on the input image and label and saves them as separate image files.

    Parameters:
        img (numpy.ndarray): The input image to be cropped and processed.
        label (numpy.ndarray): The input label image to be cropped and processed.
        height (int): The height of each crop.
        width (int): The width of each crop.
        name (str): The name of the image without the extension (to be used for saving the files).
        stride (int): The stride or step size for cropping.
        directories (list): A list of directories for saving the images and labels.
    """
    
    max_x = int(((img.shape[0]-height)/stride)+1)
    max_y = int(((img.shape[1]-width)/stride)+1)
    max_crops = (max_x)*(max_y)
    i = 0
    for h in range(max_x):
        for w in range(max_y):
                crop_img = img[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                crop_label = label[h * stride:(h * stride) + height,w * stride:(w * stride) + width]
                
                img_path = directories[1] + "/" + name + "_" + str(i+1)+".jpg"                
                mask_path = directories[2] + "/" + name + "_" + str(i+1)+".jpg"
                if len(np.unique(crop_label)) > 1: 
                    cv2.imwrite(mask_path,crop_label)
                    cv2.imwrite(img_path,crop_img)
                    i = i + 1

def patching(images_path, input_dim, stride, data, augment):  
    """
    Processes and crops images and their corresponding labels, with optional augmentation, 
    and saves the crops into specified directories.

    Parameters:
        images_path (list[str]): List of file paths to the images.
        input_dim (int): The dimension for each square crop.
        stride (int): The stride or step size for cropping.
        data (str): The data type (e.g., 'train') used to create directory paths.
        augment (bool): If True, apply augmentation to the images and labels before cropping.

    This function creates necessary directories for saving crops, extracts image names,
    and for each image, reads the image and label, applies augmentation if specified, 
    performs strided cropping, and saves the resulting crops in the appropriate directories.
    """

    directories = [f'Crops',f'Crops/{data}/image',f'Crops/{data}/mask']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    # Extract names
    image_names = []
    for img in images_path:
        image_names.append(img.split("\\")[-1].split(".")[0])    
    for name in tqdm.tqdm(image_names, total=len(image_names)):
        img_path = f"Dataset/train/image/{name}.jpg"
        img = cv2.imread(img_path)
        label_path = f"Dataset/train/mask/{name}.jpg"
        label = cv2.imread(label_path)
        if augment:
            imgs, labels = augment_data(img, label)
            for idx, (i,m) in enumerate(zip(imgs, labels)):
                strided_crop(i, m, input_dim, input_dim,f"{name}_{augment_list[idx]}", stride, directories)
        strided_crop(img, label, input_dim, input_dim, name, stride, directories) 

    
        