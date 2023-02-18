# import opencv
import cv2

# import numpy
import numpy as np

# import matplotlib
#import matplotlib.pyplot as plt

# I needs to be obtained, A needs to be obtained, invA needs to be obtained

# Input image I
I = cv2.imread('Image1.png')

# a 3x3 transform matrix A
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# the inverse of A
invA = np.linalg.inv(A)

# Height and width of the image I
height, width = I.shape[:2]
# The syntax means that the first two elements of the shape of the image I are height and width, and the third element is the number of channels
# print(height, width)

# Corners/constraints of the image I normalized
c1 = np.array([[1,1,1]]).T
c2 = np.array([[width,1,1]]).T
c3 = np.array([[1,height,1]]).T
c4 = np.array([[width,height,1]]).T

# The transform matrix*constraints, the correspondence of the four corners of the image I
cp1 = np.dot(A,c1)
cp2 = np.dot(A,c2)
cp3 = np.dot(A,c3)
cp4 = np.dot(A,c4)

# The x and y coordinates of the four corners of the image I
xp1 = cp1[0]
yp1 = cp1[1]
xp2 = cp2[0]
yp2 = cp2[1]
xp3 = cp3[0]
yp3 = cp3[1]
xp4 = cp4[0]
yp4 = cp4[1]

# The minimum and maximum x and y coordinates of the four corners of the image I
# Top left corner, Bottom left corner, Top right corner, Bottom right corner

Ap = np.array(min([1,xp1,xp2,xp3,xp4]), min([1,yp1,yp2,yp3,yp4]))
Bp = np.array(min([1,xp1,xp2,xp3,xp4]), max([yp1,yp2,yp3,yp4]))
Cp = np.array(max([xp1,xp2,xp3,xp4]), min([1,yp1,yp2,yp3,yp4]))
Dp = np.array(max([xp1,xp2,xp3,xp4]), max([yp1,yp2,yp3,yp4]))

# The minimum and maximum x and y coordinates of the four corners of the image I
minx = Ap[0], miny = Bp[1]
maxx = Cp[0], maxy = Dp[1]
 
# The meshgrid is a matrix of x and y coordinates
# It basically creates a grid of x and y coordinates
# The first parameter is the x coordinates, the second parameter is the y coordinates
[Xprime, Yprime] = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))

# So now we have our pprimematrix
pprimematrix = np.array




