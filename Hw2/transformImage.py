import cv2
import numpy as np

# Load the input image
color = cv2.imread("Image1.png", cv2.IMREAD_COLOR)
I = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Define the rotation angle
alpha = 0.5

# Define the translation vector
tx = 50
ty = 50


# Define the homography matrix A
A = np.array([[ np.cos(alpha),  -np.sin(alpha),  tx],
     [np.sin(alpha)       ,  np.cos(alpha)      , ty],
     [-0.        , -0.        ,  1.]])


# Get the dimensions of the input image
H, W = I.shape[:2]

# Define the four corners of the input image
c1 = np.array([1, 1, 1]).reshape(3, 1)
c2 = np.array([W, 1, 1]).reshape(3, 1)
c3 = np.array([1, H, 1]).reshape(3, 1)
c4 = np.array([W, H, 1]).reshape(3, 1)

# Transform the corners using the homography matrix A
cp1 = np.dot(A, c1)
cp2 = np.dot(A, c2)
cp3 = np.dot(A, c3)
cp4 = np.dot(A, c4)

# Extract the x and y coordinates of the transformed corners
xp1, yp1, _ = cp1.ravel()
xp2, yp2, _ = cp2.ravel()
xp3, yp3, _ = cp3.ravel()
xp4, yp4, _ = cp4.ravel()

# Find the minimum and maximum x and y values of the transformed corners
minx = min(1, xp1, xp2, xp3, xp4)
miny = min(1, yp1, yp2, yp3, yp4)
maxx = max(W, xp1, xp2, xp3, xp4)
maxy = max(H, yp1, yp2, yp3, yp4)

# Create a grid of x and y coordinates in the output image
Xprime, Yprime = np.meshgrid(np.arange(minx, maxx+1), np.arange(miny, maxy+1))

# Create a matrix of homogenized coordinates in the output image
heightIprime, widthIprime = Xprime.shape
pprimematrix = np.vstack((Xprime.ravel(), Yprime.ravel(), np.ones(heightIprime*widthIprime)))

# Invert the homography matrix to map points from the output image to the input image
invA = np.linalg.inv(A)

print(np.abs(np.linalg.det(invA) - 1))

# Check if the matrix is an affine transformation matrix
if np.abs(np.linalg.det(invA) - 1) > 1e-6:
    raise ValueError("Matrix is not a proper affine transformation matrix")

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

# Display the input and output images
cv2.imwrite("Input_Image.jpg", I)
cv2.imwrite("Output_Image.jpg", Iprime)
print('Transformed image saved to disk.')




