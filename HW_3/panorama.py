# The following code is for creating panoramas from a set of images
# It uses opencv and numpy
from transformImage import transformImage # From the previous homework
import cv2
import numpy as np

#from matplotlib import pyplot as plt

# HELP BLOCK to explain the code
# There are several functions defined
# Function definitions:
# 

# Once you have your photographs, it is time to obtain correspondences. Load both
# your images into MATLAB. For this assignment, you can convert them to grayscale. Also make
# sure you convert them to double using im2double.

# resize image to be divisible by 3
def resizeImage(img, scale_percent):
    # Get the current dimensions of the image
    height, width = img.shape[:2]

    # Calculate the new dimensions of the image based on the given scale
    new_height = int(height * scale_percent / 100)
    new_width = int(width * scale_percent / 100)

    # Calculate the nearest width that is divisible by 3
    new_width = int(new_width / 3) * 3

    # Calculate the nearest height that is divisible by 3
    new_height = int(new_height / 3) * 3

    # Resize the image using OpenCV's resize function
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_img

# Part 1: We set up the images we have by loading them and converting them to grayscale
def setupImages(im1, im2, scale_percent=100):
    # load images and convert to grayscale
    im1 = cv2.imread(im1, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # Resize the images to be divisible by 3
    im1 = resizeImage(im1, scale_percent)
    im2 = resizeImage(im2, scale_percent)

    # check if images are divisible by 3
    if im1.shape[0] % 3 != 0 or im1.shape[1] % 3 != 0 or im2.shape[0] % 3 != 0 or im2.shape[1] % 3 != 0:
        print("ERROR: Images are not divisible by 3")
        exit()

    # convert to double
    # im1 = im1.astype(np.float64)
    # im2 = im2.astype(np.float64)
    return im1, im2


# Obtaining Correspondences using OpenCV's ORB and Brute Force Matcher
def getCorrespondences(im1, im2):

    # Initiate ORB detector
    orb = cv2.ORB_create(nfeatures=10000)
    # find the keypoints and descriptors with ORB (https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html)
    kp1, desc1 = orb.detectAndCompute(im1,None)
    kp2, desc2 = orb.detectAndCompute(im2,None)

    # Draw keypoints on both images
    im1 = cv2.drawKeypoints(im1, kp1, None)
    im2 = cv2.drawKeypoints(im2, kp2, None)

    # Match descriptors from both images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1,desc2)

    # Extract objects for the matched points and convert them to coordinates
    im1_points = []
    im2_points = []
    for match in matches:
        im1_points.append(kp1[match.queryIdx].pt)
        im2_points.append(kp2[match.trainIdx].pt)
    

    # Draw matches between images
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)

    # Display the results
    print( "Total Keypoints for 1: {}".format(len(kp1)) )
    print( "Total Keypoints for 2: {}".format(len(kp2)) )

    cv2.imwrite("img.jpg", im1)
    cv2.imwrite("img2.jpg", im2)
    cv2.imwrite("Matches.jpg", img3)

 
    # Convert the points objects to coordinates (one row = one point[x,y])
    im1_points = np.array(im1_points)
    im2_points = np.array(im2_points)
    return im1_points, im2_points

# def getCorrespondences(im1, im2):
#     # Initiate SIFT detector
#     sift = cv2.xfeatures2d.SIFT_create(nfeatures=10000)

#     # find the keypoints and descriptors with SIFT
#     kp1, desc1 = sift.detectAndCompute(im1, None)
#     kp2, desc2 = sift.detectAndCompute(im2, None)

#     # Draw keypoints on both images
#     im1 = cv2.drawKeypoints(im1, kp1, None)
#     im2 = cv2.drawKeypoints(im2, kp2, None)

#     # Match descriptors from both images
#     bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#     matches = bf.match(desc1, desc2)

#     # Extract objects for the matched points and convert them to coordinates
#     im1_points = []
#     im2_points = []
#     for match in matches:
#         im1_points.append(kp1[match.queryIdx].pt)
#         im2_points.append(kp2[match.trainIdx].pt)

#     # Draw matches between images
#     img3 = cv2.drawMatches(im1, kp1, im2, kp2, matches, None)

#     # Display the results
#     print("Total Keypoints for 1: {}".format(len(kp1)))
#     print("Total Keypoints for 2: {}".format(len(kp2)))

#     cv2.imwrite("img.jpg", im1)
#     cv2.imwrite("img2.jpg", im2)
#     cv2.imwrite("Matches.jpg", img3)

#     # Convert the points objects to coordinates (one row = one point[x,y])
#     im1_points = np.array(im1_points)
#     im2_points = np.array(im2_points)

#     return im1_points, im2_points

    

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


def estimateTransformRANSAC(pts1, pts2):
    Nransac = 10000
    th = 5
    k = 4
    n = pts1.shape[0]
    nkeepmax = 0

    #print size of pts1 and pts2
    print('pts1 shape:', pts1.shape)
    print('pts2 shape:', pts2.shape)

    if n < k:
        print('Error: Input arrays must have at least {} data points'.format(k))
        return None

    keepmax = None

    for ir in range(Nransac):
        idx = np.random.choice(n, size=k, replace=False)
        pts1s = pts1[idx, :] # select k points from pts1
        pts2s = pts2[idx, :]

        H = estimateTransform(pts1s, pts2s)

        # pts1estim_h = np.hstack((pts1, np.ones((n, 1)))) @ H.T # homogenous coordinates
        # shape of pts1

        #python
        ones = np.ones((n, 1))
        print('pts1 shape:', pts1.shape)
        pts1_h = np.hstack((pts1,ones)) # homogenous coordinates
        # sizes
        print('pts1_h shape:', pts1_h.shape)
        pts2estim_h = np.dot(H, pts1_h.T).T # transform pts1 to pts2 using the estimated transformation matrix H
        # divide the first two rows by the third row
        # pts2estim = np.divide(pts2estim_h[:2, :], pts2estim_h[2, :]) #euclidean coordinates
        pts2estim = pts2estim_h[:, :2] / pts2estim_h[:, 2:]
        # make all pts2estim values positive
        pts2estim = np.absolute(pts2estim)
        
        # transpose the pts2estim matrix
        #pts2estim = pts2estim.T

        # shapes
        print('n:', n)
        print('pts1_h shape:', pts1_h.shape)
        print('pts2estim_h shape:', pts2estim_h.shape)
        print('pts2estim shape:', pts2estim.shape)
        print('pts2 shape:', pts2.shape)
        # print values
        print('pts2estim',pts2estim[0])
        print('pts2',pts2[0])
        
        d = np.sum((pts2estim - pts2)**2, axis=1) # euclidean distance

        keep = np.where(d < th)[0]
        # print('keep:', keep)
        nkeep = len(keep)
        # print("length: ",len(keep))

        if nkeep > nkeepmax:
            nkeepmax = nkeep
            Hkeepmax = H
            keepmax = keep

    if keepmax is None:
        print('Error: RANSAC failed to find a suitable transform')
        return None

    pts1keep = pts1[keepmax,:]
    pts2keep = pts2[keepmax,:]

    Hbetter = estimateTransform(pts1keep, pts2keep)

    return Hbetter



# This gives us a hypothesis for a homography matrix between two sets of correspondence points, it is called in RANSAC repeatedly
def estimateTransform(pts1, pts2):
    n = pts1.shape[0]
    A = np.zeros((2*n, 9))

    for i in range(n):
        x = pts1[i, 0]
        y = pts1[i, 1]
        xp = pts2[i, 0]
        yp = pts2[i, 1]

        A[2*i, :] = [-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp]
        A[2*i+1, :] = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]

    _, _, V = np.linalg.svd(A)

    H = np.reshape(V[-1, :], (3, 3))

    # Force the homography matrix to satisfy the constraints
    # H = H / H[2, 2]
    # H /= H[2, 2]
    # H = np.clip(H, -10, 10)
    # print(H)

    return H # we return a normalized homography matrix, the last value is 1
    

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



def main():
    im1 = 'images\Image1.jpg'
    im2 = 'images\Image2.jpg'

    test1 = np.array([[1373, 1204], [1841, 1102], [1733, 1213], [2099, 1297]])
    # print(test1.shape)
    test2 = np.array([[182, 1160], [728, 1055], [617, 1172], [1001, 1247]])

    #H= estimateTransform(test1, test2)
    #print(H)

    image1, image2 = setupImages(im1, im2)
    im1_points, im2_points = getCorrespondences(image1, image2)
    #print("im1_points",im1_points)
    #print("im2_points",im2_points)
    # size print
    print("size of im1_points",im1_points.shape)
    print("size of im2_points",im2_points.shape)

    Hv2 = estimateTransformRANSAC(im1_points, im2_points)
    #inverse of Hv2
    Hv2inv = np.linalg.inv(Hv2)
    print("Hv2inv",Hv2inv)

    transformImage(image2, Hv2inv,"homography")

    # size of image1
    print("size of image1",image1.shape)
    

    # moo1, moo2 = getCorrespondences(image1, image2)
 
    # homoo = estimateTransformRANSAC(moo1, moo2)
    

if __name__ == '__main__':
    main()