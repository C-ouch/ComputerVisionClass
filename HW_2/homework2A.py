import cv2
import numpy as np


# compute the inverse of a translation matrix
def inv_translation_matrix(tx, ty):
    return np.array([[1, 0, -tx],
                     [0, 1, -ty],
                     [0, 0, 1]])

# compute the inverse of a rotation matrix
def inv_rotation_matrix(alpha):
    return np.array([[np.cos(alpha), np.sin(alpha), 0],
                     [-np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 1]])

# compute the inverse of a shear matrix using a standard matrix inversion formula for a 2 by 2 matrix
def inv_shear_matrix(alpha):
    return np.array([[1, -np.tan(alpha), 0],
                     [0, 1, 0],
                     [0, 0, 1]])
