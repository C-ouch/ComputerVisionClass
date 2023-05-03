# Alif Jakir and Evan Couchman
# Assignment 5 - Deep Learning for Computer Vision on CIFAR-10

Training a deep learning network on the CIFAR-10 dataset, which is a dataset of 32x32 color images in 10 classes, however our network is using a datset of 8 classes. The train/validate/test split is (0.7/0.2/0.1)

We tried to use grid search on the hyperparameters of the network to find the best hyperparameters, with the following hyperparameters:

learning_rate_values = [0.01, 0.001, 0.0001]
batch_size_values = [32, 64, 128]

This unfortunately resulted in not testing multiple each hp individually but continued to train one large model with a changing learning rate and batch size. Which can be seen in the graph below. It took several hours of running the network over all the combinations to get to this conclusion. Here it is:

![log_acc_lr_0 0001_batch_128](https://user-images.githubusercontent.com/67016155/235348778-f3a445ca-6baf-4415-ab9b-4d1080a13b44.png)

If you check the .png's in the /HW_5/Images_GS directory, we have several graphs of accuracy over time and ones for training loss for different combinations of hyperparameters during grid search.

After that we used a brute forced method only testing 4 of the cominations from the 6 hps and 1 extra outside those learning rates: 

Hyperparameter Combinations: 
1. learning rate: 0.1 | Batch Size: 64
2. learning rate: 0.01 | Batch Size: 32 (First hps for training using Grid search so accuracy and loss is still valid)
3. learning rate: 0.01 | Batch Size: 64
4. learning rate: 0.001 | Batch Size: 64
5. learning rate: 0.0001 | Batch Size: 64
6. learning rate: 0.0001 | Batch Size: 128

These models .png's can be found in the /HW_5/Images directory, with the graphs for accuracy over time and training loss. After brute forcing different values its shown that a learning rate of 0.1 caused the training to be come static very early making it invalidated and that the combinaton (learning rate: 0.001, batch size: 64) caused the accuracy after epoch 3 to decline after reaching its peak. The best hyperparameter combination was (learning rate: 0.001, batch size: 64) which resulted in an final acuracy of 81.22%. This models accuracy and loss graph can be seen below:
 
 ![log_acc_lr_0 001_batch_64](https://github.com/Caerii/CS473-ComputerVisionClass/blob/main/HW_5/Images/log_acc_lr_0.001_batch_64.png) ![log_loss_lr_0 001_batch_64](https://github.com/Caerii/CS473-ComputerVisionClass/blob/main/HW_5/Images/log_train_lr_0.001_batch_64.png)

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

