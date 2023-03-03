# The following code is for creating panoramas from a set of images
# It uses opencv and numpy
from transformImage import transformImage # From the previous homework
import cv2
import numpy as np

# HELP BLOCK to explain the code
# There are several functions defined
# Function definitions:
# 

# Once you have your photographs, it is time to obtain correspondences. Load both
# your images into MATLAB. For this assignment, you can convert them to grayscale. Also make
# sure you convert them to double using im2double.  
def setupImages(im1, im2):
    # load images and convert to grayscale
    im1 = cv2.imread(im1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # convert to double
    im1 = im1.astype(np.float64)
    im2 = im2.astype(np.float64)
    return im1, im2



# Write a function, estimateTransform to determine the transform between ‘im1_points’
# to ‘im2_points’, i.e., to determine the transform between ‘im1’ to ‘im2’. Your function should
# have the form:
# A=estimateTransform( im1_points, im2_points );

# In class, we saw how to estimate a 3 ×3 homography
# A =
# a b c
# d e f
# g h i
#
# by setting
# q = [a b c d e f g h i]T
# and creating a design matrix P and a vector r. We also saw how to estimate q by using homo-
# geneous least squares, with singular value decomposition (SVD). The function estimateTransform
# should create P and r according to the direct linear transform (DLT) approach we saw in class.
# Note: you will get full credit for using a for loop to create P. However, there are ways to create
# both P without using a for loop. While r is relatively easy to create without a for loop, you
# will get 1 bonus point for figuring out how to create P without using a for loop! Once you
# create P, use the SVD-based method we discussed in class to obtain q from P. Then rear-
# range the values of q to get the values in A. For the following set of points in ‘im1=Image1’
# and‘im2=Image2’:
# m1_points =
# 1373 1204
# 1841 1102
# 1733 1213
# 2099 1297
# 
# and im2_points =
# 182 1160
# 728 1055
# 617 1172
# 1001 1247
#
# A should be:
# −0.0004272, −0.0002588, 0.8379231
# −0.0000576, −0.0007065, 0.5457875
# −0.0000000, −0.0000003, 0.0001091
#
# or
# 
# 0.0004272, 0.0002588, −0.8379231
# 0.0000576, 0.0007065, −0.5457875
# 0.0000000, 0.0000003, −0.0001091
#
# i.e., where the second solution is the negative of the first solution (like scale, the negation
# of the homography matrix does not impact the result). Use the points and A listed above
# only to verify your function estimateTransform. However, you must report results for your
# own set of points, not the ones listed here. Unless you use points obtained using SURF with
# RANSAC as discussed below, you will not get full credit.
# Note: In MATLAB, you will get a stubby matrix of size 8 ×9, in which case you should use
# the regular version of the SVD function. If your matrix were 9 ×9 or taller, then you would use
# the flag ‘econ’ with the SVD function.
# In class, we looked at using RANSAC to sample random sets of 4 correspondences and get
# a near-optimal solution that throws outliers. Write a function estimateTransformRANSAC,
# that uses RANSAC to estimate the transformation between the points in ‘im1_points’ and
# ‘im2_points’. In each RANSAC iteration, you will call the function estimateTransform using
# the minimum number of correspondences k. What should k be for the homogeneous least
# squares version?
# An important consideration will be how you choose the error threshold t for RANSAC, and
# the number of RANSAC iterations Nransac. In your write-up include how you chose the error
# threshold and the number of RANSAC iterations.
# Once you perform RANSAC, you will get a list of points Nagree which is a subset of all N points.
# To balance the solution, you should perform a final homogeneous least squares solve on all
# Nagree points, by calling estimateTransform on the set Nagree points. Your assignment must
# demonstrate that you have performed this final solve to receive full points.
# Note: While we looked at using regular least squares to estimate q using the pseudo-inverse of
# P, that method is numerically ill-conditioned. Ifyouimplementthepseudo-inversemethod,
# you will lose all 75 points on this section! You must use SVD to get full points.
# Apply your function esimateTransformRANSAC to ‘im1_points’ and ‘im2_points’ in your workspace
# to get the transform matrix, A. You will find that when you use RANSAC, A may be different
# from the one above. This is fine, as the results from RANSAC + homogeneous least squares
# depend upon the random samples chosen.

# This gives us a homography matrix
def estimateTransform(im1_points, im2_points):
    # create P and r
    P = np.zeros((8, 9)) # 8x9 matrix that will be used to get q
    r = np.zeros((8, 1)) # 8x1 matrix that will be used to get q
    for i in range(4):
        x1 = im1_points[i][0]
        y1 = im1_points[i][1]
        x2 = im2_points[i][0]
        y2 = im2_points[i][1]
        P[2 * i] = [0, 0, 0, -x1, -y1, -1, y2 * x1, y2 * y1, y2]
        P[2 * i + 1] = [x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2]
        r[2 * i] = y2
        r[2 * i + 1] = x2

    # use SVD to get q
    U, s, V = np.linalg.svd(P)
    q = V[-1]

    # rearrange q to get A
    A = np.array([[q[0], q[1], q[2]], [q[3], q[4], q[5]], [q[6], q[7], q[8]]])
    return A

# 4: Applying the Homography
# Use the function transformImage written in Assignment 1 to transform ‘im2’ to
# match ‘im1’. Call the transformed image ‘im2_transformed’.
# Be careful! A relates ‘im1’ to ‘im2’. To transform ‘im2’ to ‘im1’, you have to apply the in-
# verse of A to ‘im2’!
# For the panorama, you will find it easier to force the corners in the transformed image to
# be at (1,1) instead of (minx,miny).
# Also, interp2 may provide NaN (not a number) values. You can reset NaNs to zeros by calling
# nanlocations = isnan( im2_transformed );
# im2_transformed( nanlocations )=0;
# For the example images, ‘im2_transformed’ will be similar the following image:
# It is quite likely that your ‘im2_transformed’ will not appear exactly like the one shown here,
# and will have slight differences. Again, this is fine, as your ‘im2_transformed’ depends on the
# estimate of A calculated using your correspondences. It should not look too different from
# the image shown here.