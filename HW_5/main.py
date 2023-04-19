# Build our dataset
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pickle

import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, path):
        
        data = pickle.load(f)

        self.images, self.labels = data
        def __len__(self):
            return len(self.images)
        

    def __getitem__(self, key):
        image = self.images[index]
        label = self.labels[index]
        label = int(label)

        return (image, label)

    path = "data_demo.pk1"

    dataset = MyDataset(path)
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

            def forard(self, x):
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

    lr = 0.001

    # optimizer: adam
    optimizer = Adam(params = net.parameters(), lr = lr)

    # Define my loss function
    loss_fn = nn.NNLLLoss()

    def train(epoch):
        # Define my loss function

        # start loop here

        # get the data

        # Pass the data to the network to get some predictions

        # 