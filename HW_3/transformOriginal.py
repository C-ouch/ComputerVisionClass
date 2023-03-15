import cv2
import numpy as np

# # Load the input image
# color = cv2.imread("Image1.png", cv2.IMREAD_COLOR)
# I = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


# Define the homography matrix A
def homography_matrix(A, B, C, D, E, F, G, H, I):
    M = np.array([[A, B, C], [D, E, F], [G, H, I]])
    return M


def transform_matrix(tx=None, ty=None, alpha=None, sx=None, sy=None, shear_factor=None):
    if alpha is not None:
        # Rotation matrix
        R = np.array([[np.cos(np.deg2rad(alpha)), -np.sin(np.deg2rad(alpha)), 0],
                      [np.sin(np.deg2rad(alpha)), np.cos(np.deg2rad(alpha)), 0],
                      [0, 0, 1]])
    else:
        R = np.eye(3)
        
    if sx is not None or sy is not None:
        # Scaling matrix
        S = np.array([[sx if sx is not None else 1, 0, 0],
                      [0, sy if sy is not None else 1, 0],
                      [0, 0, 1]])
    else:
        S = np.eye(3)
        
    if tx is not None or ty is not None:
        # Translation matrix
        T = np.array([[1, 0, tx if tx is not None else 0],
                      [0, 1, ty if ty is not None else 0],
                      [0, 0, 1]])
    else:
        T = np.eye(3)
        
    if shear_factor is not None:
        # Shear matrix
        Sh = np.array([[1, shear_factor, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
    else:
        Sh = np.eye(3)
        
    # Compose the transformation matrix
    A = np.dot(np.dot(np.dot(T, Sh), np.dot(S, R)), np.linalg.inv(T))
    
    return A

def transformImage(I, A, transform_type, output_image_name):
    print(transform_type)
    
    # Check if the transform type is valid
    if transform_type == 'scaling'or'translation'or'rotation'or'reflection'or'homography'or'affine':

        # Get the dimensions of the input image
        H, W = I.shape[:2]
        


        # Define the four corners of the input image
        c1 = np.array([1, 1, 1]).reshape(3, 1)
        c2 = np.array([W, 1, 1]).reshape(3, 1)
        c3 = np.array([1, H, 1]).reshape(3, 1)
        c4 = np.array([W, H, 1]).reshape(3, 1)


        # Transform the corners using the homography matrix A into correspondences
        cp1 = np.dot(A, c1)
        cp2 = np.dot(A, c2)
        cp3 = np.dot(A, c3)
        cp4 = np.dot(A, c4)

        # Check if the matrix is a homography matrix
        if transform_type == ("homography"):
            cp1 = cp1[:2] / cp1[2]
            cp2 = cp2[:2] / cp2[2]
            cp3 = cp3[:2] / cp3[2]
            cp4 = cp4[:2] / cp4[2]
        

        # Extract the x and y coordinates of the transformed corners
        # if statement to check if the matrix is a homography matrix
        if transform_type == ("homography"):
            xp1, yp1 = cp1.flatten()
            xp2, yp2 = cp2.flatten()
            xp3, yp3 = cp3.flatten()
            xp4, yp4 = cp4.flatten()
        else:
            xp1, yp1, _ = cp1.flatten()
            xp2, yp2, _ = cp2.flatten()
            xp3, yp3, _ = cp3.flatten()
            xp4, yp4, _ = cp4.flatten()

        # Find the minimum and maximum x and y values of the transformed corners
        minx = min(1, xp1, xp2, xp3, xp4)
        miny = min(1, yp1, yp2, yp3, yp4)
        maxx = max(W, xp1, xp2, xp3, xp4)
        maxy = max(H, yp1, yp2, yp3, yp4)
        #print out xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4
        print(xp1, xp2, xp3, xp4, yp1, yp2, yp3, yp4)

        print("min and max: ", minx, miny, maxx, maxy)

        # Create a grid of x and y coordinates in the output image
        Xprime, Yprime = np.meshgrid(np.arange(minx, maxx+1), np.arange(miny, maxy+1))


        # Create a matrix of homogenized coordinates in the output image
        heightIprime, widthIprime = Xprime.shape
        pprimematrix = np.vstack((Xprime.flatten(), Yprime.flatten(), np.ones(heightIprime*widthIprime)))


        print("after homogenized: " , heightIprime, widthIprime)
        # Invert the homography matrix to map points from the output image to the input image
        invA = np.linalg.inv(A)


        # Map the homogenized output image coordinates to input image coordinates using the inverted homography matrix
        phatmatrix = np.dot(invA, pprimematrix)

        # Extract the x and y coordinates from the mapped homogenized input image coordinates
        xlongvector = phatmatrix[0] / phatmatrix[2]
        ylongvector = phatmatrix[1] / phatmatrix[2]

        # Reshape the x and y coordinates into a matrix
        xmatrix = xlongvector.reshape(heightIprime, widthIprime)
        ymatrix = ylongvector.reshape(heightIprime, widthIprime)

        # Interpolate the input image at the mapped coordinates to create the output image
        Iprime = cv2.remap(I, xmatrix.astype(np.float32), ymatrix.astype(np.float32), cv2.INTER_LINEAR)

        # crop the output image to fit in the original image size
        if transform_type == ("reflection"):
            Iprime = Iprime[0:H, 0:W]
      


        print("at end of func: ",Iprime.shape)
        # Display the input and output images
        cv2.imwrite("out_im\Input_Image.jpg", I)
        cv2.imwrite(f"out_im\{output_image_name}.jpg", Iprime)

        return Iprime

        print('Transformed image saved to disk.')
    else:
        raise ValueError('The transform type is not valid')