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

    return transformed_image

# The transformImage function takes an input image, a 3x3 transformation matrix, 
# and a transform type as arguments, and returns the transformed image.

# The function first gets the dimensions of the input image, 
# and defines the output image dimensions as the same as the input image. 
# It then creates a meshgrid of pixel coordinates in the output image, 
# and a matrix of homogenized coordinates in the output image.

# The function then computes the inverse transformation matrix based on the transform type, 
# using the intuitions from Section 1. If the inverse transformation matrix is an affine transformation 
# matrix, the function checks if the determinant is close enough to 1. 
# If it is not, the function raises a ValueError.

# The function then maps the homogenized output image coordinates to 
# input image coordinates using the inverse transform, and extracts the x and y coordinates 
# from the mapped homogenized input image coordinates. 
# Finally, the function interpolates the input image at the mapped coordinates to create 
# the output image, and returns



def main():
    # Read the input image
    color = cv2.imread("Image1.png", cv2.IMREAD_COLOR)

    input_image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Define the scaling factor
    scaling_factor = 0.5

    # Define the rotation angle
    rotation_angle = 30

    # Define the translation vector
    translation_vector = np.array([0, 0])

    # Define the shear factor
    shear_factor = 0.5

    # Define the transformation matrices
    affine_matrix = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    homography_matrix = np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    scaling_matrix = np.array([[scaling_factor, 0, 0], [0, scaling_factor, 0], [0, 0, 1]])
    rotation_matrix = np.array([[np.cos(np.deg2rad(rotation_angle)), -np.sin(np.deg2rad(rotation_angle)), 0], [np.sin(np.deg2rad(rotation_angle)), np.cos(np.deg2rad(rotation_angle)), 0], [0, 0, 1]])
    translation_matrix = np.array([[1, 0, translation_vector[0]], [0, 1, translation_vector[1]], [0, 0, 1]])
    #A reflection matrix that reflects the matrix in the y direction
    reflect_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])

    #reflection_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0], [0, 0, 1]])

    #Change the size of the image to 1080 ×1920 (make sure the image you start with is not already 1080 ×1920)
    #if the input image is not 1080x1920, resize it with an if condition
    if input_image.shape[0] != 1080 or input_image.shape[1] != 1920:
        input_image = cv2.resize(input_image, (1920, 1080))
  
    #Reflect the image in the y direction
    reflected_image = transformImage(input_image, reflect_y, 'reflection')

    #Rotate the image clockwise by 30 degrees
    # rotated_image = transformImage(input_image, rotation_matrix, 'rotation')

    #Shear the image in the x-direction so that the additional amount added to each x value is 0.5 times each y value
    # sheared_image = transformImage(input_image, shear_matrix, 'shear')

    #Translate the image by 300 in the x-direction and 500 in the y-direction, then rotate the resulting image counterclockwise by 20 degrees, then scale the resulting image down to one-half its size. You should apply the transformImage function only once to do this.


    # Display the input and output images
    cv2.imwrite("In_Image.jpg", input_image)
    cv2.imwrite("Refl_Image.jpg", reflected_image)
    print('Transformed image saved to disk.')

if __name__ == '__main__':
    main()