import subprocess
import sys

# Function to run a shell command
def run_command(command):
    subprocess.check_call(command, shell=True)

# Function to install a package
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install the necessary package
install("einops")
install("timm")

# Enable Jupyter nbextension
run_command("jupyter nbextension enable --py widgetsnbextension")

# Global packages
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import einops
from tqdm.notebook import tqdm
import time
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score, auc, mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomVerticalFlip, RandomHorizontalFlip, RandomCrop, RandomResizedCrop, ColorJitter, CenterCrop

from typing import Tuple
import timm
from PIL import Image

import importlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")