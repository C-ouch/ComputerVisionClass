# CIFAR-10 dataset exploration with machine learning
# By Alif Jakir and Evan Couchman
# jakirab@clarkson.edu
# 4/28/23 

# Build our dataset
# For this assignment, you will train a performant CNN-based classifier on CIFAR-10 by performing a grid search over two variables that impact network performance. Some examples of variables you can play with are:

# Learning Rate (YES)
# Batch Size (YES)
# Optimizer
# Number of network layers
# Number of convolutional kernals
# Network structure (adding more dropout layers, adding batch normalization, etc.)
# Data augmentation (cropping, standardization, etc.)

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle
import random
import os

if not os.path.exists("predictions"):
    os.mkdir("predictions")


import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path, split="train"):
        """Initialize the dataset"""
        assert split in ["train", "val", "test"]
        
        data = pickle.load(open(path,"rb")) # load the data by unpickling the file
    
        self.images, self.labels = data # unpack the tuple

        #the data needs to be shuffled because the data is in order, so we need to shuffle it
        # zip the images and labels together
        zipped = list(zip(self.images, self.labels))
        # shuffle the zipped list
        random.shuffle(zipped)
        # unzip the zipped list
        self.images, self.labels = zip(*zipped)


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
        # print(label) # ERROR WHY IS THE LABEL FORMAT NOT FOLLOWING THE PROTOCOL?
        # create first image
        # print shape of image and axes
        
        image = np.transpose(image, axes=(2,0,1)) # transpose the image by swapping the axes
        image = image.astype(np.float32) # convert the image to a float32

        # normalize the image
        image = normalize(image)

        # split the label by the first . starting from the right side of the label
        last_dot_index = label.rfind(".")
        label = int(label[last_dot_index-1])
        # print("label: ", label)

        return (image, label) # return the image and the label



def normalize(image):
    # [ 0 > 255] > [-1, 1]
    image = image / 255 # [0, 1]
    image = (image - 0.5) / 0.5 # [-1, 1]
    return image

def unnormalize(image):
    image = (image + 1) # [0, 1]
    image = (image - 0.5) / 0.5 # [-1, 1]
    return image

import torch.nn as nn
import torch

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(4*4*128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


from torch.optim import Adam

path = "HW_5\data\data.pkl" # dataset path

dataset = MyDataset(path) # create the dataset
image, label = dataset[0]

net = MyNet()

# Hyperparameters

lr_values = [0.01, 0.001, 0.0001]
batch_size_values = [32, 64, 128]
# lr = 0.01 # learning rate
# batch_size = 128 # batch size
num_epochs = 10 # number of epochs
model_path = "model.pt"

# create a device object
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define my loss function
loss_fn = nn.NLLLoss()

# logging variables
log_train = []
log_val = []
log_acc = []


def train(epoch):
    net.train()
    for ii,(images, labels) in enumerate(dataloader):
        optimizer.zero_grad() # zero the gradients in order to prevent them from accumulating

        images, labels = images.to(device), labels.to(device) # move the data to the device
        # pass the data to the network to get some predictions
        predictions = net(images)

        # calculate the loss
        loss = loss_fn(predictions, labels)

        # backpropagate the loss
        loss.backward()

        # update the weights
        optimizer.step() 

        # add the loss to the training log
        log_train.append(loss.item())

        # print the loss
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

        # calculate the accuracy
        # acc = accuracy(predictions, labels)
        # print("Epoch: {}, Accuracy: {}".format(epoch, acc))

def validate(epoch):
    net.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    for images, labels in dataloader_val:

        # move images and labels to the cpu/gpu
        images, labels = images.to(device), labels.to(device)

        # predict stuff
        predictions = net(images)

        # get the loss
        loss = loss_fn(predictions, labels)

        # preds: (batch_size, 10)
        predictions_as_labels = predictions.argmax(1) # convert the predictions to labels

        # get number of correct predictions
        num_correct = (predictions_as_labels == labels).sum().item()
        num_batch = len(labels)
        total_correct += num_correct
        total_samples += num_batch

        # do some logging
        log_val.append(loss.item())

        # do backprop
        total_loss += loss.item()
    
    images, labels, predictions = images.cpu(), labels.cpu(), predictions.cpu()
    visualize_predictions(epoch, images, labels, predictions)

    log_acc.append(total_correct / total_samples)
    print("Validation loss for epoch {}: {} with accuracy {} / {}".format(
        epoch, total_loss, total_correct, total_samples
    ))

def visualize_predictions(epoch, images, labels, predictions):
    # images: (batch_size, 3, 32, 32)
    # labels: (batch_size)
    # predictions: (batch_size, 10)

    # detach from graph
    predictions = predictions.detach()

    # convert to numpy
    images = images.numpy()
    labels = labels.numpy()
    predictions = predictions.numpy()

    # (batch_size, 10) > (batch_size)
    predictions = predictions.argmax(1)

    # iterate over all our samples and save with plt
    for ii in range(len(predictions)):
        image = images[ii, :, :, :]
        prediction = predictions[ii]
        label = labels[ii]

        # convert images c w h > w h c
        image = np.transpose(image, (1, 2, 0)) # (3, 32, 32) > (32, 32, 3)
        image = unnormalize(image) # unnormalize the image because we normalized it before
        image = image.astype(np.uint8) # convert to uint8 because plt needs it

        # save
        plt.imshow(image)
        plt.savefig(f"predictions/{epoch}_{ii}_{label}_{prediction}.png")

# in pytorch, you need to save optimizer, and your model

# see if model exists, if it does, load it and continue training
cur_epoch = 0
# if os.path.exists(model_path):
#     nstate_dict, ostate_dict, log_train, log_val, log_acc, cur_epoch = torch.load(model_path)
#     # Load the params for our network and optimizer
#     net.load_state_dict(nstate_dict)
#     optimizer.load_state_dict(ostate_dict)
# if the model doesn't exist, train from scratch    



for lr in lr_values:
    for batch_size in batch_size_values:
        # create dataloaders with the current hyperparameters
        dataloader = DataLoader(MyDataset(path, split="train"), batch_size=batch_size)
        dataloader_val = DataLoader(MyDataset(path, split="val"), batch_size=batch_size)

        # create the optimizer with the current learning rate
        optimizer = Adam(params=net.parameters(), lr=lr)

        # train and validate the model with the current hyperparameters
        for epoch in range(num_epochs):
            train(epoch)
            validate(epoch)

            print("Saving the model we have... beep boop")
            torch.save(
                [
                    net.state_dict(), optimizer.state_dict(),
                    log_train, log_val, log_acc,
                    epoch + 1,
                ],
                model_path
            )

        # Make some plots from our logs
        plt.clf()
        plt.plot(log_train, label="train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.savefig(f"log_train_lr_{lr}_batch_{batch_size}.png")

        plt.clf()
        plt.plot(log_acc)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.savefig(f"log_acc_lr_{lr}_batch_{batch_size}.png")


