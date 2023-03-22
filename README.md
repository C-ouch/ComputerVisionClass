# ComputerVisionClass CS473 Spring 2023
Here are ways that we made computers see!

Homework 2 - Here we have some transforms applied to some images.

We have defined a function that takes in a matrix and does some transform on an image.

Homework 3 - Inside of /out_im/ are the images that were created by the program.

The way that the panorama works is we first find the keypoints and descriptors for each image. Then we match the keypoints and descriptors between the two images. Then we find the homography between the two images. We apply the homography to the second image. Then we stitch the images together using a blending ramp.