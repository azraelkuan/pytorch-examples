import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from skimage import io


TEST_DATA_ROOT = "../data/testData/"


def load_data():
    file_names = os.listdir(TEST_DATA_ROOT)
    print(file_names)



if __name__ == '__main__':
    load_data()





