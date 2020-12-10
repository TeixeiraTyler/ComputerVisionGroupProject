import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from CNN import ConvNet


# Trains and tests model, and outputs losses and accuracies
class Trainer:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def train(self, model, device, train_loader, optimizer, criterion, epoch, batch_size,
                indices, train_losses, train_accuracies):
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

        # Set model to train before each epoch
        model.train()

        # Empty list to store losses
        losses = []
        correct = 0

        # Iterate over entire training samples (1 epoch)
        for batch_idx, batch_sample in enumerate(train_loader):
            data, target = batch_sample
            #print(target)

            # Push data/label to correct device
            data, target = data.to(device), target.to(device)

            # Reset optimizer gradients. Avoids grad accumulation.
            optimizer.zero_grad()

            # Do forward pass for current set of data
            output = model(data)
            
            # Compute loss based on criterion
            loss = criterion(output, target)

            # Computes gradient based on final loss
            loss.backward()

            # Store loss
            losses.append(loss.item())

            # Optimize model parameters
            optimizer.step()

            # Get predicted index by rounding the sigmoid output to 0 or 1
            pred = torch.round(output)

            # Count correct predictions overall
            correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate and print training loss and accuracy
        train_loss = float(np.mean(losses))
        train_acc = (100. * correct) / ((batch_idx + 1) * batch_size * 40)
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            float(np.mean(losses)), correct, (batch_idx + 1) * batch_size * 40,
                                             100. * correct / ((batch_idx + 1) * 40 * batch_size)))

        # Track total training losses and accuracies per epoch for later plotting
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        indices.append(epoch)

        return train_loss, train_acc

    def test(self, model, device, test_loader, indices, test_losses, test_accuracies):
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
        criterion = nn.BCEWithLogitsLoss()

        # Set torch.no_grad() to disable gradient computation and backpropagation
        with torch.no_grad():
            for batch_idx, sample in enumerate(test_loader):
                data, target = sample
                data, target = data.to(device), target.to(device)

                # Predict for data by doing forward pass
                output = model(data)

                # Compute loss based on same criterion as training
                loss = criterion(output, target)

                # Append loss to overall test loss
                losses.append(loss.item())

                # Get predicted index by rounding the sigmoid output to 0 or 1
                pred = torch.round(output)

                # Count correct predictions overall
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Calculate and print training loss and accuracy
        test_loss = float(np.mean(losses))
        accuracy = (100. * correct) / (40*(len(test_loader.dataset)))
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, 40*(len(test_loader.dataset)), accuracy))

        # Track total testing losses and accuracies per epoch for later plotting
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        return test_loss, accuracy

    def go(self, dataset1, dataset2):
        # Check if cuda is available
        use_cuda = torch.cuda.is_available()

        # Set proper device based on cuda availability
        device = torch.device("cuda" if use_cuda else "cpu")
        print("Torch device selected: ", device)

        # Set indices and other variables to plot losses and accuracies
        indices = []
        train_losses = []
        train_accuracies =[]
        test_losses = []
        test_accuracies = []
        
        # Initialize the model and send to device
        model = ConvNet().to(device)

        # Initialize the criterion for loss computation
        criterion = nn.BCEWithLogitsLoss()

        # Initialize optimizer type
        optimizer = optim.Adam(model.parameters())

        train_loader = DataLoader(dataset1,
                                    batch_size=self.FLAGS.batch_size,
                                    shuffle=True, num_workers=4)
        test_loader = DataLoader(dataset2,
                                    batch_size=self.FLAGS.batch_size,
                                    shuffle=False, num_workers=4)

        best_accuracy = 0.0

        # Run training for n_epochs specified in config
        for epoch in range(1, self.FLAGS.num_epochs + 1):
            print("\nEpoch: ", epoch)
            train_loss, train_accuracy = self.train(model, device, train_loader,
                                               optimizer, criterion, epoch, self.FLAGS.batch_size,
                                               indices, train_losses, train_accuracies)
            test_loss, test_accuracy = self.test(model, device, test_loader,
                                                indices, test_losses, test_accuracies)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

        # Print best accuracy results
        print("accuracy is {:2.2f}".format(best_accuracy))
        print("Training and evaluation finished")

        # Plot training and testing data
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('CelebA Training and Testing Results')
        ax1.set(xlabel='Epoch', ylabel='Loss')
        ax1.plot(indices, train_losses, color='blue')
        ax1.plot(indices, test_losses, color='red')
        ax2.set(xlabel='Epoch', ylabel='Accuracy')
        ax2.plot(indices, train_accuracies, color='blue')
        ax2.plot(indices, test_accuracies, color='red')
        plt.show()
