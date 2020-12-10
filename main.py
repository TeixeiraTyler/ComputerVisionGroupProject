import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CNN import ConvNet
from Trainer import Trainer

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=10,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')

    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()

    # Create transformations to apply to each data sample
    # Can specify variations such as image flip, color flip, random crop, ...
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Receive dataset of images
    dataset1 = datasets.CelebA('.\\', split='train', download=False,
                              transform=transform)
    dataset2 = datasets.CelebA('.\\', split='test', download=False,
                              transform=transform)

    # Retype dataset for compatability; subsample it for faster runtime
    dataset1.attr = dataset1.attr.type(dtype=torch.FloatTensor)
    dataset2.attr = dataset1.attr.type(dtype=torch.FloatTensor)
    dataset1 = torch.utils.data.Subset(dataset1, range(0, 100000, 100))
    dataset2 = torch.utils.data.Subset(dataset2, range(0, 10000, 20))

    # Train network
    trainer = Trainer(FLAGS)
    trainer.go(dataset1, dataset2)