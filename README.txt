Instructions (w/ Anaconda & Python 3.8):
1. Open your conda environment.
2. Ensure that you've installed all the necessary libraries (See below).
3. Navigate to project directory.
4. run 'python main.py'

These steps can also be followed when not using Anaconda. Simply use your terminal.
Other versions of Python may also work.

Additional arguments (optional):
--num_epochs (default 10)
--batch_size (default 10)
--log_dir (default 'logs')

WARNING: RUN TIME MAY BE EXCESSIVELY LONG WITH THE FULL DATASET
2 commented out lines exist in main.py that create a subsample of the dataset. These can be uncommented for use.
Lines 43 & 44

Needed libraries (using most stable versions as of December 10, 2020):
torch
torchvision
argparse
matplotlib
numpy

Code from Programming Assignment 1 part 2 was used as a foundation for the training, testing, and design of our CNN