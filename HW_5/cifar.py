# For this assignment, you will train a performant CNN-based classifier on CIFAR-10 by performing a grid search over two variables that impact network performance. Some examples of variables you can play with are:

# Learning Rate
# Batch Size
# Optimizer
# Number of network layers
# Number of convolutional kernals
# Network structure (adding more dropout layers, adding batch normalization, etc.)
# Data augmentation (cropping, standardization, etc.)

# We will be using pytorch

# Importing the libraries
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
import sys
import argparse

# Setting the seed
torch.manual_seed(0)

# Defining the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Defining the hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_kernals', type=int, default=32)
parser.add_argument('--network_structure', type=str, default='CNN')
parser.add_argument('--data_augmentation', type=str, default='None')
args = parser.parse_args()

# Defining the transforms
if args.data_augmentation == 'None':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
elif args.data_augmentation == 'Standardization':
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
elif args.data_augmentation == 'Crop':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
elif args.data_augmentation == 'Standardization_Crop':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# Loading the dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# Defining the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Defining the network



