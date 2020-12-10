import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Define various layers here, such as in the tutorial example
        self.conv1 = nn.Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=5,
                                stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, 
                                out_channels=128,
                                kernel_size=5,
                                stride=2)

        self.fc_m2 = nn.Linear(640, 100)

        self.fc_m4 = nn.Linear(100, 100)

        self.fc1 = nn.Linear(15360, 10000)
        self.fc2 = nn.Linear(10000, 8000)
        self.fc3 = nn.Linear(8000, 40)

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
        # ======================================================================
        # One hidden layer

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

    # Use two convolutional layers.
    def model_2(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.

        X = torch.sigmoid(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = torch.sigmoid(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = torch.sigmoid(self.fc_m2(X))
        X = torch.sigmoid(self.fc2(X))

        return X

    # Replace sigmoid with relu.
    def model_3(self, X):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with relu.

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m2(X))
        X = F.relu(self.fc2(X))

        return X

    # Add one extra fully connected layer.
    def model_4(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with relu.

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m2(X))
        X = F.relu(self.fc_m4(X))
        X = F.relu(self.fc2(X))
        X = F.relu(X)

        return X

    # Use Dropout now.
    def model_5(self, X):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with relu.
        # and  + Dropout.

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc_m5_a(X))
        X = self.dropout1(X)
        X = F.relu(self.fc_m5_b(X))
        X = self.dropout1(X)
        X = F.relu(self.fc_m5_c(X))

        return X
