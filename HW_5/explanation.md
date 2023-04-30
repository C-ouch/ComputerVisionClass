# Alif Jakir and Evan Couchman
# Assignment 5 - Deep Learning for Computer Vision on CIFAR-10

Training a deep learning network on the CIFAR-10 dataset, which is a dataset of 32x32 color images in 10 classes, however our network is using a datset of 8 classes. The train/validate/test split is (0.7/0.2/0.1)

We use grid search on the hyperparameters of the network to find the best hyperparameters. We use the following hyperparameters:

learning_rate_values = [0.01, 0.001, 0.0001]
batch_size_values = [32, 64, 128]

If you check the .png's in the /HW_5 directory, we have several graphs of training accuracy over time for different combinations of hyperparameters.

Interestingly enough we have found that the learning rate of .0001 and with a batch size of 128 converges to 70% accuracy within 10 epochs, so it is better than any other combination of hyperparameters. It took several hours of running the network over all the combinations to get to this conclusion. Here it is:

![log_acc_lr_0 0001_batch_128](https://user-images.githubusercontent.com/67016155/235348778-f3a445ca-6baf-4415-ab9b-4d1080a13b44.png)


# Network Architecture

The network architecture consists of two main parts: convolutional layers and fully connected layers. The convolutional layers use filters to extract features from the input images, and the fully connected layers are used to classify the features extracted from the convolutional layers. Specifically, there are three sets of convolutional layers, each followed by a max pooling layer to reduce the spatial dimensions of the output. The fully connected layers consist of two linear layers with a ReLU activation function and a dropout layer between them. The final layer uses a logarithmic softmax activation function to produce class probabilities.

# How to load the model

If you want to load the "model.pt" file in the root directory, you basically use this snippet of code:

```python
model_path = "model.pt"

if os.path.exists(model_path):
     nstate_dict, ostate_dict, log_train, log_val, log_acc, cur_epoch = torch.load(model_path)
     # Load the params for our network and optimizer
     net.load_state_dict(nstate_dict)
     optimizer.load_state_dict(ostate_dict)
```

# How to run training
Just run the following commands in the root directory:

```python
python -m pip install requirements.txt

python main.py
```

It will automatically run the grid search and train the network with all hyperparameter configurations. It will save the last model trained with its hyperparameters to the root directory as "model.pt"

# Visualization of model predictions that were incorrect or correct

