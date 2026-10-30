import os
import time
 
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
 
from modules import UNet2D               
from dataset import HipMRIDataset

EST_IMG_DIR = "keras_slices_test"
TEST_SEG_DIR = "keras_slices_seg_test"
WEIGHTS_PATH = "unet2d.pth"
OUT_DIR = "predictions"
NUM_CLASSES = 6                           
BATCH_SIZE = 8
NUM_FIGURES = 3                            # how many best / worst slices to plot
EPS = 1e-6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

