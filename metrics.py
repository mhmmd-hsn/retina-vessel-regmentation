import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def iou(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) metric.

    The IoU metric is used to evaluate the performance of image segmentation 
    models. It is calculated as the ratio of the intersection area to the union 
    area between the ground truth (y_true) and the predicted (y_pred) segmentations.

    Parameters:
        y_true: Tensor of ground truth binary masks.
        y_pred: Tensor of predicted binary masks.

    Returns:
        A tensor containing the IoU score as a float32 value.
    """

    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

smooth = 1e-15
def dice_coef(y_true, y_pred):
    """
    Computes the Dice Coefficient between the ground truth (y_true) and the predicted (y_pred) binary masks.

    The Dice Coefficient is a measure of overlap between two binary masks. It is calculated as 2 * (intersection / (total elements in first mask + total elements in second mask)).

    Parameters:
        y_true: Tensor of ground truth binary masks.
        y_pred: Tensor of predicted binary masks.

    Returns:
        A tensor containing the Dice Coefficient as a float32 value.
    """
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
