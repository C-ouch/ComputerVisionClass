# The following code is for creating panoramas from a set of images
# It uses opencv and numpy
from transformOriginal import transformImage # From the previous homework
import cv2
import numpy as np

from matplotlib import pyplot as plt

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

    # cv2.imwrite("out_im\img.jpg", im1) #uncomment to see corresponding keypoints
    # cv2.imwrite("out_im\img2.jpg", im2)
    # cv2.imwrite("out_im\Matches.jpg", img3)

 
    # Convert the points objects to coordinates (one row = one point[x,y])
    im1_points = np.array(im1_points)
    im2_points = np.array(im2_points)
    return im1_points, im2_points

def estimateTransformRANSAC(pts1, pts2):
    """Estimate the transform between two sets of points using RANSAC"""
    # RANSAC stands for Random Sample Consensus
    Nransac = 1000 # number of RANSAC iterations
    th = 5 # threshold for the distance between a point and the estimated line
    k = 4 # number of points to fit a model
    n = pts1.shape[0] # number of points
    nkeepmax = 0 # number of points that agree with the model

    #print size of pts1 and pts2
    print('pts1 shape:', pts1.shape)
    print('pts2 shape:', pts2.shape)

    if n < k: # if there are less points than the minimum required
        print('Error: Input arrays must have at least {} data points'.format(k))
        return None

    keepmax = None # indices of the points that agree with the model

    for ir in range(Nransac): # for each RANSAC iteration
        idx = np.random.choice(n, size=k, replace=False)
        pts1s = pts1[idx, :] # select k points from pts1
        pts2s = pts2[idx, :]

        H = estimateTransform(pts1s, pts2s)

        ones = np.ones((n, 1)) # column of ones
        pts1_h = np.hstack((pts1,ones)) # homogenous coordinates

        pts2estim_h = np.dot(H, pts1_h.T).T # transform pts1 to pts2 using the estimated transformation matrix H
        # divide the first two rows by the third row
        pts2estim = pts2estim_h[:, :2] / pts2estim_h[:, 2:]
        # make all pts2estim values positive
        pts2estim = np.absolute(pts2estim)
        
        d = np.sum((pts2estim - pts2)**2, axis=1) # euclidean distance
        d = np.sqrt(d)

        keep = np.where(d < th)[0]
        nkeep = len(keep)

        if nkeep > nkeepmax: # if we have more inliers than before, update the best model
            nkeepmax = nkeep # update the number of inliers
            Hkeepmax = H # keep the best model
            keepmax = keep # keep the indices of the inliers

    if keepmax is None: # if we didn't find any inliers
        print('Error: RANSAC failed to find a suitable transform')
        return None

    pts1keep = pts1[keepmax,:] # keep the inliers
    pts2keep = pts2[keepmax,:] # keep the inliers

    Hbetter = estimateTransform(pts1keep, pts2keep) # perform a final least squares solve on the inliers

    return Hbetter

# This gives us a hypothesis for a homography matrix between two sets of correspondence points, it is called in RANSAC repeatedly
def estimateTransform(pts1, pts2):
    """Estimate a homography matrix between two sets of points."""
    n = pts1.shape[0] # number of points
    A = np.zeros((2*n, 9)) # create an empty matrix A

    for i in range(n): # for each point, we create a row in A
        x = pts1[i, 0] # x coordinate of the point in image 1
        y = pts1[i, 1] # y coordinate of the point in image 1
        xp = pts2[i, 0] # x coordinate of the point in image 2
        yp = pts2[i, 1] # y coordinate of the point in image 2

        # the A matrix is filled in with the values of the points
        A[2*i, :] = [-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp] # fill in the even rows of A
        A[2*i+1, :] = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp] # fill in the odd rows of A

    
    # create P, which is the design matrix
    P = np.zeros((2*n, 9))
    # create P without a for loop by using numpy
    # it works by dividing the last column of A by the last value of each row of A
    P = A / A[:, -1][:, np.newaxis]

    # create r in order to use SVD
    # r is the last column of A
    r = np.zeros((2*n, 1)) # create an empty matrix r
    r[:, 0] = A[:, 8] # fill in the last column of A

    # create q from P based on Singular Value Decomposition method
    # get P transpose
    PTP = np.dot(P.T, P) # A symmetric matrix that contains the covariance of the data, used as intermediate step in Principal Component Analysis
    eigenvalues, eigenvectors = np.linalg.eig(PTP) # perform SVD on P by using eigenvalues and eigenvectors
    min_eigenvalue_idx = np.argmin(eigenvalues) # it is important to get the smallest eigenvalue
    q = eigenvectors[:, min_eigenvalue_idx] # q is the last column of V, it is the eigenvector corresponding to the smallest eigenvalue

    # rearrange the values of q to get the values in A
    A = np.zeros((3, 3))
    A[0, :] = q[0:3]
    A[1, :] = q[3:6]
    A[2, :] = q[6:9]

    return A # return the estimated homography matrix

def blendImages(im1, im2, ramp1_start, ramp2_start, ramp_gradient=0.6):
    """Blend two images together using ramp blending.
    The images might be of different sizes, but the blending region will be the same.
    ramp1_start: the x value for the start of the ramp in the first image
    ramp2_start: the x value for the start of the ramp in the second image
    The ramp gradient is the slope of the ramp.
    """

    # Create empty ramp arrays for both images
    ramp1 = np.zeros(im1.shape[1])
    ramp2 = np.zeros(im2.shape[1])

    # Create the ramps
    ramp1[:ramp1_start] = np.linspace(1, 0, ramp1_start)
    ramp2[:ramp2_start] = np.linspace(1, 0, ramp2_start)

    ramp = ramp1 + ramp2 # Combine the ramps
    ramp = ramp / ramp_gradient # Normalize the ramp

    # Apply the ramp to im1 and im2
    im1_blend = im1 * ramp[np.newaxis, :] # np.newaxis is used to increase the dimension of the existing array by one more dimension
    im2_blend = im2 * (1 - ramp)[np.newaxis, :]

    # Combine the blended images
    panorama = im1_blend + im2_blend
    # write the intermediate files
    cv2.imwrite('HW_3\out_im\im1_blend.png', im1_blend)
    cv2.imwrite('HW_3\out_im\im2_blend.png', im2_blend)

    return panorama # type: np.ndarray

def main():
    im1 = 'HW_3\images\Room1.jpg'
    im2 = 'HW_3\images\Room2.jpg'

    image1, image2 = setupImages(im1, im2) # load the images and convert them to grayscale
    im1_points, im2_points = getCorrespondences(image1, image2) # get the corresponding points
    Hv2 = estimateTransformRANSAC(im1_points, im2_points) # estimate the homography matrix
    Hv2inv = np.linalg.inv(Hv2) # calculate inverse of Hv2
    im2_transformed = transformImage(image2, Hv2inv,"homography", 'out_im\im2_transformed.png') # transform the second image
    im1_expanded = np.zeros_like(im2_transformed) # Expand the first image to the size of the second image
    im1_expanded[:image1.shape[0], :image1.shape[1]] = image1 # copy the first image into the expanded image
    #instead of being on the top left, the image is on the bottom left
    #Realignment for set 1 (Image): -608
    #Realignment for set 2 (Room): -557
    #Realignment for set 3 (Bathroom): -320
    im1_expanded = np.roll(im1_expanded, int(-image1.shape[0]-557), axis=0)

    # take the input from the user for the start of both ramps
    # manually locate two points in image 1 using ginput from matplotlib
    plt.imshow(im1_expanded)
    x_1 = plt.ginput(1, timeout=0)
    ramp1_start = int(x_1[0][0])
    plt.imshow(im2_transformed)
    x_2 = plt.ginput(1, timeout=0)
    #print it out
    print("x_1",x_1)
    print("x_2",x_2)
    ramp2_start = int(x_2[0][0])
    ramp_gradient = 0.99
    # panorama = blendImages(im1_expanded, im2_transformed, 2780, 2500, 0.99)
    panorama = blendImages(im1_expanded, im2_transformed, ramp1_start, ramp2_start, ramp_gradient)

    # Save the intermediate and final images
 
    cv2.imwrite('HW_3\out_im\im1_expanded.jpg', im1_expanded)
    cv2.imwrite('HW_3\out_im\im2_transformed.jpg', im2_transformed)
    cv2.imwrite('HW_3\out_im\panorama_set1.jpg', panorama)


if __name__ == '__main__':
    main()