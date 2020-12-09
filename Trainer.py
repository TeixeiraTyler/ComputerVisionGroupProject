from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
# from ConvNet import ConvNet
import numpy as np
from CNN import ConvNet

# import face_recognition


class Trainer:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    # BELOW IS TO BE MODIFIED. GRABBED FROM ANSWER TO PROGRAMMING ASSIGNMENT 1 PART 2.

    def train(self, model, device, train_loader, optimizer, criterion, epoch, batch_size):
        '''
        Trains the model for an epoch and optimizes it.
        model: The model to train. Should already be in correct device.
        device: 'cuda' or 'cpu'.
        train_loader: dataloader for training samples.
        optimizer: optimizer to use for model parameter updates.
        criterion: used to compute loss for prediction and target
        epoch: Current epoch to train for.
        batch_size: Batch size to be used.
        '''

        # Set model to train mode before each epoch
        model.train()

        # Empty list to store losses
        losses = []
        correct = 0

        # Iterate over entire training samples (1 epoch)
        for batch_idx, batch_sample in enumerate(train_loader):
            data, target = batch_sample

            # Push data/label to correct device
            data, target = data.to(device), target.to(device)

            # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
            optimizer.zero_grad()

            # Do forward pass for current set of data
            output = model(data)

            # ======================================================================
            # Compute loss based on criterion
            loss = criterion(output, target)

            # Computes gradient based on final loss
            loss.backward()

            # Store loss
            losses.append(loss.item())

            # Optimize model parameters based on learning rate and gradient
            optimizer.step()

            # Get predicted index by selecting maximum log-probability
            pred = output.argmax(dim=1, keepdim=True)

            # ======================================================================
            # Count correct predictions overall
            correct += pred.eq(target.view_as(pred)).sum().item()

        train_loss = float(np.mean(losses))
        train_acc = correct / ((batch_idx + 1) * batch_size)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            float(np.mean(losses)), correct, (batch_idx + 1) * batch_size,
                                             100. * correct / ((batch_idx + 1) * batch_size)))
        return train_loss, train_acc

    def test(self, model, device, test_loader):
        '''
        Tests the model.
        model: The model to train. Should already be in correct device.
        device: 'cuda' or 'cpu'.
        test_loader: dataloader for test samples.
        '''

        # Set model to eval mode to notify all layers.
        model.eval()

        losses = []
        correct = 0

        # Set torch.no_grad() to disable gradient computation and backpropagation
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                data, target = sample
                data, target = data.to(device), target.to(device)

                # Predict for data by doing forward pass
                output = model(data)

                # ======================================================================
                # Compute loss based on same criterion as training
                loss = F.cross_entropy(output, target, reduction='mean')

                # Append loss to overall test loss
                losses.append(loss.item())

                # Get predicted index by selecting maximum log-probability
                pred = output.argmax(dim=1, keepdim=True)

                # ======================================================================
                # Count correct predictions overall
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss = float(np.mean(losses))
        accuracy = 100. * correct / len(test_loader.dataset)

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset), accuracy))

        return test_loss, accuracy

    def go(self, dataset1, dataset2):
        # Check if cuda is available
        use_cuda = torch.cuda.is_available()

        # Set proper device based on cuda availability
        device = torch.device("cuda" if use_cuda else "cpu")
        print("Torch device selected: ", device)

        # Initialize the model and send to device
        model = ConvNet(self.FLAGS.mode).to(device)

        # Initialize the criterion for loss computation
        criterion = nn.CrossEntropyLoss(reduction='mean')

        # Initialize optimizer type
        optimizer = optim.SGD(model.parameters(), lr=self.FLAGS.learning_rate, weight_decay=1e-7)

        train_loader = DataLoader(dataset1, batch_size=self.FLAGS.batch_size,
                                  shuffle=True, num_workers=4)
        test_loader = DataLoader(dataset2, batch_size=self.FLAGS.batch_size,
                                 shuffle=False, num_workers=4)

        best_accuracy = 0.0

        # Run training for n_epochs specified in config
        for epoch in range(1, self.FLAGS.num_epochs + 1):
            print("\nEpoch: ", epoch)
            train_loss, train_accuracy = self.train(model, device, train_loader,
                                               optimizer, criterion, epoch, self.FLAGS.batch_size)
            test_loss, test_accuracy = self.test(model, device, test_loader)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

        print("accuracy is {:2.2f}".format(best_accuracy))

        print("Training and evaluation finished")
