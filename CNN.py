import torch
import torch.nn as nn
import torch.nn.functional as F

# Contains all usable models and their corresponding layers
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # Conv, fc, and dropout layers defined here
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=32,
                                kernel_size=5,
                                stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, 
                                out_channels=64,
                                kernel_size=5,
                                stride=2)

        self.fc1 = nn.Linear(7680, 9000)
        self.fc2 = nn.Linear(9000, 7000)
        self.fc3 = nn.Linear(7000, 40)

        self.dropout1 = nn.Dropout2d(0.5)

        # Use the selected model below
        self.forward = self.model

    def model(self, X):
        '''
        2 convolutional layers and 2 fully connected layers.
        ReLU, sigmoid, max pooling, and dropout involved.
        Final layer has 40 sigmoid outputs, representing the multi-class
        binary classification involved for each attribute
        '''

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        X = self.dropout1(X)
        X = F.relu(self.fc2(X))
        X = self.dropout1(X)
        X = torch.sigmoid(self.fc3(X))

        return X
