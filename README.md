# ComputerVisionClass CS473 Spring 2023
Here are ways that we made computers see!

Homework 2 - Here we have some transforms applied to some images.

We have defined a function that takes in a matrix and does some transform on an image.

Homework 3 - Inside of HW_3/out_im/ are the images that were created by the program. Here are the panorama images:


Step 1: Get the first image to be expanded as such

![im1_expanded](https://user-images.githubusercontent.com/67016155/226788645-13fcaf7a-1345-496f-bb64-00a7236adb9e.jpg)

Step 2: Apply the transform to the second image

![im2_transformed](https://user-images.githubusercontent.com/67016155/226788685-c8c0d2a5-4e9b-4ee7-a0a6-f3f7ee67a02f.jpg)

Step 3: Apply blend

![im1_blend](https://user-images.githubusercontent.com/67016155/226788775-94e52cf7-0151-42cd-86b5-1caba5aec31a.png)
![im2_blend](https://user-images.githubusercontent.com/67016155/226788800-d4241e5d-eff5-4483-9bb5-173ae82c4d18.png)

Step 4: Add together (most excellently)

![panorama_set1](https://user-images.githubusercontent.com/67016155/226788574-de2636c8-f098-4e09-824d-82e870c0694d.jpg)

![panorama_set1](https://user-images.githubusercontent.com/67016155/226788549-638b1ca1-9d8b-48e9-b879-a7402e460d7e.jpg)

![panorama_set1](https://user-images.githubusercontent.com/67016155/226788526-6be40f2d-62c7-4f39-b9ac-5acb18eb4b79.jpg)


The way that the panorama works is we first find the keypoints and descriptors for each image. Then we match the keypoints and descriptors between the two images. Then we find the homography between the two images. We apply the homography to the second image. Then we stitch the images together using a blending ramp.

Homework 4 -

MATLAB Code for Camera Calibration Pose Estimation

Homework 5 - Training a deep learning network on the CIFAR-10 dataset, which is a dataset of 32x32 color images in 10 classes, however our network is using a datset of 8 classes. The train/validate/test split is (0.7/0.2/0.1)

We use grid search on the hyperparameters of the network to find the best hyperparameters. We use the following hyperparameters:

learning_rate_values = [0.01, 0.001, 0.0001]
batch_size_values = [32, 64, 128]

If you check the .png's in the /HW_5 directory, we have several graphs of training accuracy over time for different combinations of hyperparameters.

Interestingly enough we have found that the learning rate of .0001 and with a batch size of 128 converges to 70% accuracy within 10 epochs, so it is better than any other combination of hyperparameters. It took several hours of running the network over all the combinations to get to this conclusion.