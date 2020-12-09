import face_recognition
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from CNN import ConvNet
from Recognizer import Recognizer
from Trainer import Trainer

# Set parameters for Sparse Autoencoder
parser = argparse.ArgumentParser('CNN Exercise.')
parser.add_argument('--mode',
                    type=int, default=1,
                    help='Select mode between 1-5.')
parser.add_argument('--learning_rate',
                    type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--num_epochs',
                    type=int,
                    default=60,
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

image = face_recognition.load_image_file('./img/image1.jpg')
recognizer = Recognizer(image)
recognizer.go()

# Receive dataset of images
# dataset1 = datasets.CelebA('./data/', split='train', download=True,
#                           transform=transform)
# dataset2 = datasets.CelebA('./data/', split='test', download=True,
#                           transform=transform)

# Train network
# trainer = Trainer(FLAGS)
# trainer.go(dataset1, dataset2)

# Show/plot results
plt.plot()
# plt.show()
