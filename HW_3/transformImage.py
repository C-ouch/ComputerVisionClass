import cv2
import numpy as np

def transformImage(input_image, transform_matrix, transform_type):
    # Get the dimensions of the input image
    hin, win = input_image.shape[:2]

    # Define the output image dimensions
    hout, wout = hin, win

    # Create a meshgrid of pixel coordinates in the output image
    xout, yout = np.meshgrid(np.arange(wout), np.arange(hout))

    # Create a matrix of homogenized coordinates in the output image
    p_out = np.vstack((xout.flatten(), yout.flatten(), np.ones(hout*wout)))

    # Compute the inverse transformation matrix based on the transform type
    if transform_type == 'scaling':
        inv_transform = np.diag([1/transform_matrix[0,0], 1/transform_matrix[1,1], 1])
    elif transform_type == 'rotation':
        inv_transform = np.linalg.inv(transform_matrix)
    elif transform_type == 'translation':
        inv_transform = np.eye(3)
        inv_transform[0:2,2] = -transform_matrix[0:2,2]
    elif transform_type == 'reflection':
        inv_transform = np.linalg.inv(transform_matrix)
        # Apply the transformation to the input image
        # This does a reflection on the y-axis
        # transformed_image = cv2.warpAffine(input_image, inv_transform[0:2], (win, hin))
        return transformed_image
    elif transform_type == 'shear':
        inv_transform = np.eye(3)
        inv_transform[0,1] = -transform_matrix[0,1]/transform_matrix[0,0]
    elif transform_type == 'affine':
        # Check if the inverse transformation matrix is an affine transformation matrix
        if np.abs(np.linalg.det(inv_transform) - 1) > 1e-6:
            print(np.abs(np.linalg.det(inv_transform) - 1))
            raise ValueError('Matrix is not a proper affine transformation matrix')
        inv_transform = np.linalg.inv(transform_matrix)
    elif transform_type == 'homography':
        inv_transform = np.linalg.inv(transform_matrix)
    else:
        raise ValueError('Invalid transform type specified')

    

    # Map the homogenized output image coordinates to input image coordinates using the inverse transform
    p_in = np.dot(inv_transform, p_out)

    # Extract the x and y coordinates from the mapped homogenized input image coordinates
    xin = p_in[0] / p_in[2]
    yin = p_in[1] / p_in[2]

    # Interpolate the input image at the mapped coordinates to create the output image
    transformed_image = cv2.remap(input_image, xin.reshape(hout, wout).astype(np.float32), yin.reshape(hout, wout).astype(np.float32), cv2.INTER_LINEAR)

    # write the image to file
    cv2.imwrite('transformed_image.png', transformed_image)

    return transformed_image