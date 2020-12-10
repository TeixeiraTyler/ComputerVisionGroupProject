import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# Contains all usable models and their corresponding layers
class ConvNet(nn.Module):
    def __init__(self, mode):
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

        #self.forward = self.model_1

        # Choose which CNN model to use
        if mode == 1:
            self.forward = self.model_1
        elif mode == 2:
            self.forward = self.model_2
        elif mode == 3:
            self.forward = self.model_3
        elif mode == 4:
            self.forward = self.model_4
        elif mode == 5:
            self.forward = self.model_5
        else: 
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)


    # BELOW ARE THE MODELS FROM THE ANSWER TO PROGRAMMING ASSIGNMENT 1 PART 2

    # The main model I am testing. Similar to model_5 from original assignment. Ignore model_2 through model_5
    def model_1(self, X):
        '''
        2 convolutional layers and 2 fully connected layers.
        ReLU, sigmoid, max pooling, and dropout involved.
        Final layer has 40 sigmoid outputs, representing the multi-class
        binary classification involved for each attribute
        '''

        X = F.relu(self.conv1(X))
        #print(X.size())
        X = F.max_pool2d(X, 2)
        #print(X.size())
        X = F.relu(self.conv2(X))
        #print(X.size())
        X = F.max_pool2d(X, 2)
        #print(X.size())
        X = torch.flatten(X, start_dim=1)
        #print(X.size())
        X = F.relu(self.fc1(X))
        #print(X.size())
        X = self.dropout1(X)
        #print(X.size())
        X = F.relu(self.fc2(X))
        #print(X.size())
        X = self.dropout1(X)
        #print(X.size())
        X = torch.sigmoid(self.fc3(X))
        #print(X.size())

        return X