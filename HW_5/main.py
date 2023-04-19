# CIFAR-10 dataset exploration with machine learning

# Build our dataset
# For this assignment, you will train a performant CNN-based classifier on CIFAR-10 by performing a grid search over two variables that impact network performance. Some examples of variables you can play with are:

# Learning Rate
# Batch Size
# Optimizer
# Number of network layers
# Number of convolutional kernals
# Network structure (adding more dropout layers, adding batch normalization, etc.)
# Data augmentation (cropping, standardization, etc.)

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle

import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path, split="train"):
        """Initialize the dataset"""
        assert split in ["train", "val", "test"]
        
        data = pickle.load(open(path,"rb")) # load the data by unpickling the file

        self.images, self.labels = data # unpack the tuple

        # train val test ratio of 0.7/0.2/0.1
        if split == "train":
            self.images = self.images[: int(0.7 * len(self.images))]
            self.labels = self.labels[: int(0.7 * len(self.labels))]
        elif split == "val":
            self.images = self.images[int(0.7 * len(self.images)) : int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.7 * len(self.labels)) : int(0.8 * len(self.labels))]
        elif split == "test":
            self.images = self.images[int(0.8 * len(self.images)) :]
            self.labels = self.labels[int(0.8 * len(self.labels)) :]

    def __len__(self): # return the length of the dataset
        return len(self.images) # return the length of the dataset
        
    def __getitem__(self, key): # return the item at the given key
        
        image = self.images[key] # get the image
        label = self.labels[key] # get the label
        print(label) # ERROR WHY IS THE LABEL FORMAT NOT FOLLOWING THE PROTOCOL?
        # create first image
        image = np.transpose(image, axes=(2,0,1)) # transpose the image by swapping the axes
        
        image = image.astype(np.float32) # convert the image to a float32
        
        # label format:
        # ...-label-...
    
        # label_split = label.split("-") # split the label by the dash
        # label = label_split[1] # get the label
        # # label = int(label) # convert the label to an integer
        # label = int(label.split("_")[0])

        return (image, label) # return the image and the label

# path = "./data/data.pkl" # path to the dataset
path = "HW_5\data\data.pkl"

dataset = MyDataset(path) # create the dataset
image, label = dataset[0]

import torch.nn as nn
import torch

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # conv > relu > max > conv > relu > max > fc (10) > softmax

        # input: (32, 32, 3)
        self.conv1 = nn.Conv2d(
            in_channels = 3, 
            out_channels = 32,
            kernel_size = 5,
        ) # output: (30, 30, 32)

        self.maxpool1 = nn.MaxPool2d(
            kernel_size = 2,
        )

        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
        ) # output: (15, 15, 64)

        self.maxpool2 = nn.MaxPool2d(
            kernel_size = 2,
        ) # output: (6,6, 64)

        self.linear= nn.Linear(
            in_features = 6*6*64,
            out_features = 10,
        )

        self.softmax = nn.LogSoftmax(
            dim = None,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        # output: (6,6, 64)

        x = torch.flatten(x, 1)

        x = self.linear(x)
        x = self.softmax(x)

        return x

from torch.optim import Adam

net = MyNet()

# Hyperparameters

lr = 0.01 # learning rate
batch_size = 128 # batch size
num_epochs = 10 # number of epochs

# optimizer: adam
optimizer = Adam(params = net.parameters(), lr = lr)

# Define my loss function
loss_fn = nn.NLLLoss()


# Create dataloader to handle batching
dataloader = DataLoader(dataset, batch_size = batch_size)

def accuracy(predictions, labels):
    with torch.no_grad():
        _, predicted = torch.max(predictions, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        acc = correct / total
        return acc

def test():
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            predictions = net(images)
            _, predicted = torch.max(predictions.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    return acc


def train(epoch):
    net.train()
    for images, labels in dataloader:
        optimizer.zero_grad() # zero the gradients in order to prevent them from accumulating
        # pass the data to the network to get some predictions
        predictions = net(images)

        # calculate the loss
        loss = loss_fn(predictions, labels)

        # backpropagate the loss
        loss.backward()

        # update the weights
        optimizer.step() 

        # print the loss
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

        # calculate the accuracy
        acc = accuracy(predictions, labels)
        print("Epoch: {}, Accuracy: {}".format(epoch, acc))

for epoch in range(num_epochs):
    # save the model for each epoch
    train(epoch)
    acc = test()
    torch.save(net.state_dict(), "model_best.pt")