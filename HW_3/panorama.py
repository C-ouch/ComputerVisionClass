# The following code is for creating panoramas from a set of images
# It uses opencv and numpy
from transformOriginal import transformImage # From the previous homework
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

    cv2.imwrite("out_im\img.jpg", im1)
    cv2.imwrite("out_im\img2.jpg", im2)
    cv2.imwrite("out_im\Matches.jpg", img3)

 
    # Convert the points objects to coordinates (one row = one point[x,y])
    im1_points = np.array(im1_points)
    im2_points = np.array(im2_points)
    return im1_points, im2_points




def estimateTransformRANSAC(pts1, pts2):
    """Estimate the transform between two sets of points using RANSAC"""
    # RANSAC is Ra
    Nransac = 1000
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

    keepmax = None # indices of the points that agree with the model

    for ir in range(Nransac): # for each RANSAC iteration
        idx = np.random.choice(n, size=k, replace=False)
        pts1s = pts1[idx, :] # select k points from pts1
        pts2s = pts2[idx, :]

        H = estimateTransform(pts1s, pts2s)

        ones = np.ones((n, 1))
        # print('pts1 shape:', pts1.shape)
        pts1_h = np.hstack((pts1,ones)) # homogenous coordinates
        # sizes
        # print('pts1_h shape:', pts1_h.shape)
        pts2estim_h = np.dot(H, pts1_h.T).T # transform pts1 to pts2 using the estimated transformation matrix H
        # divide the first two rows by the third row
        # pts2estim = np.divide(pts2estim_h[:2, :], pts2estim_h[2, :]) #euclidean coordinates
        pts2estim = pts2estim_h[:, :2] / pts2estim_h[:, 2:]
        # make all pts2estim values positive
        pts2estim = np.absolute(pts2estim)

        # shapes
        # print('n:', n)
        # print('pts1_h shape:', pts1_h.shape)
        # print('pts2estim_h shape:', pts2estim_h.shape)
        # print('pts2estim shape:', pts2estim.shape)
        # print('pts2 shape:', pts2.shape)
        # # print values
        # print('pts2estim',pts2estim[0])
        # print('pts2',pts2[0])
        
        d = np.sum((pts2estim - pts2)**2, axis=1) # euclidean distance

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


# blend two images together using the ramp blur blending method
# def blendImages(im1, im2, blur_threshold=1):
#     """Blend two images together using ramp blur blending."""
#     # get the size of the images
#     im1_size = im1.shape
#     im2_size = im2.shape



#     # create the ramp blur
#     ramp = np.zeros(im1_size)
#     for i in range(im1_size[0]):
#         for j in range(im1_size[1]):
#             if (i < im1_size[0] * blur_threshold):
#                 ramp[i, j] = 1
#             else:
#                 ramp[i, j] = (im1_size[0] - i) / (im1_size[0] * (1 - blur_threshold))

#     # blend the images together
#     blended = im1 * ramp + im2 * (1 - ramp)

#     return blended, ramp

# def blendImages(im1, im2, blend_width = 100):
#     """Blend two images together using a linear ramp."""
#     h, w = im1.shape[:2]

#     # Create the ramp
#     ramp = np.zeros(w)
#     ramp[:blend_width] = np.linspace(0, 1, blend_width)
#     ramp[-blend_width:] = np.linspace(1, 0, blend_width)

#     # Apply the ramp to im1 and im2
#     im1_blend = im1 * ramp[np.newaxis, :]
#     im2_blend = im2 * (1 - ramp)[np.newaxis, :]

#     #write out images of blended images
#     cv2.imwrite('out_im\im1_blend.jpg', im1_blend)
#     cv2.imwrite('out_im\im2_blend.jpg', im2_blend)


#     # Combine the blended images
#     panorama = im1_blend + im2_blend

#     return panorama

""" def blendImages(im1, im2, blend_width=100, vertical_blend=False):
    #Blend two images together using a linear ramp.

    def find_overlap(im1, im2, blend_width):
        

        result = cv2.matchTemplate(im1, im2, cv2.TM_CCOEFF_NORMED)
        _,_,_, max_loc = cv2.minMaxLoc(result)

        if vertical_blend:
            overlap_start = max_loc[1]
            overlap_end = overlap_start + im2.shape[0]
        else:
            overlap_start = max_loc[0]
            overlap_end = overlap_start + im2.shape[1]

        return overlap_start, overlap_end

    h, w = im1.shape[:2]

    # Find the overlapping region
    overlap_start, overlap_end = find_overlap(im1, im2, blend_width)

    if vertical_blend:
        ramp = np.zeros(h)
        ramp[overlap_start:overlap_end] = np.linspace(0, 1, overlap_end - overlap_start)
        ramp[:overlap_start] = 0
        ramp[overlap_end:] = 1

        # Apply the ramp to im1 and im2
        im1_blend = im1 * (1 - ramp)[:, np.newaxis]
        im2_blend = im2 * ramp[:, np.newaxis]

    else:
        ramp = np.zeros(w)
        ramp[overlap_start:overlap_end] = np.linspace(0, 1, overlap_end - overlap_start)
        ramp[:overlap_start] = 0
        ramp[overlap_end:] = 1

        # Apply the ramp to im1 and im2
        im1_blend = im1 * (1 - ramp)[np.newaxis, :]
        im2_blend = im2 * ramp[np.newaxis, :]

    # Combine the blended images
    panorama = im1_blend + im2_blend

    return panorama """

def blendImages(im1, im2):
    # Load two images
    img1 = im1
    img2 = im2

    #shape of the image1
    print("im1:",img1.shape)

    # Create a ramp function
    ramp = np.linspace(0, 1, img1.shape[1]).reshape((1, img1.shape[1], 1))
    ramp_reshaped = ramp.reshape(1, img1.shape[1], 1)

    #print ramp and ramp_reshaped
    print("ramp:",ramp.shape)
    print("ramp_reshaped:",ramp_reshaped.shape)

    # Create an array of ones with the same shape as img1
    ones = np.ones_like(img1)

    
    # Create the inverse ramp function
    inv_ramp = np.flip(ramp_reshaped, axis=1)

    # Multiply the first image by the ramp function
    img1 = cv2.multiply(img1, ramp_reshaped)

    # Multiply the second image by the inverse ramp function
    img2 = cv2.multiply(img2, inv_ramp)

    # Add the two images together
    result = cv2.add(img1, img2)

    # Display the result
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    im1 = 'images\Room1.jpg'
    im2 = 'images\Room2.jpg'

    test1 = np.array([[1373, 1204], [1841, 1102], [1733, 1213], [2099, 1297]])
    # print(test1.shape)
    test2 = np.array([[182, 1160], [728, 1055], [617, 1172], [1001, 1247]])

    #H= estimateTransform(test1, test2)
    #print(H)

    image1, image2 = setupImages(im1, im2)
    im1_points, im2_points = getCorrespondences(image1, image2)

    Hv2 = estimateTransformRANSAC(im1_points, im2_points)
    #inverse of Hv2
    Hv2inv = np.linalg.inv(Hv2)
    print("Hv2inv",Hv2inv)

    im2_transformed = transformImage(image2, Hv2inv,"homography", 'out_im\im2_transformed.png')
    #print shape of im2_transformed
    print("im2_transformed",im2_transformed.shape)


    # Expand the first image
    im1_expanded = np.zeros_like(im2_transformed)
    #print shaoe of im1_expanded
    print("im1_expanded",im1_expanded.shape)

    im1_expanded[:image1.shape[0], :image1.shape[1]] = image1
    #instead of being on the top left, the image is on the bottom left
    #Realignment for set 1: -608
    #Realignment for set 2: -557
    #Realignment for set 3: -320
    im1_expanded = np.roll(im1_expanded, int(-image1.shape[0]-557), axis=0)

    blend_width = 100

    panorama = blendImages(im1_expanded, im2_transformed)
    # Blending the images with 0.3 and 0.7
    # panorama = cv2.addWeighted(im1_expanded, 0.3, im2_transformed, 0.7, 0)

    print("panorama",panorama.shape)

    cv2.imwrite('out_im\panoramav4.jpg', panorama)

    
    

    # # Create the ramp
    # h, w = im1_expanded.shape[:2]
    # ramp = np.zeros(w)
    # ramp[:blend_width] = np.linspace(0, 1, blend_width)
    # ramp[-blend_width:] = np.linspace(1, 0, blend_width)

    # Save the intermediate and final images
    # cv2.imwrite('out_im\im2_transformed.png', im2_transformed)
    cv2.imwrite('out_im\im1_expanded.png', im1_expanded)
    # print("shape of exp*ramp", (im1_expanded * ramp[np.newaxis, :]).shape)

    # cv2.imwrite('out_im\im1_blend.png', im1_expanded * ramp[np.newaxis, :])
    # #print hello
    # print("hello")
    # cv2.imwrite('out_im\im2_blend.png', im2_transformed * (1 - ramp)[np.newaxis, :])
    cv2.imwrite('out_im\panorama.png', panorama)

if __name__ == '__main__':
    main()