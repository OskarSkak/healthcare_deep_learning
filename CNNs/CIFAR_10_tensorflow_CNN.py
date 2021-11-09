import numpy as np
import pyreadr
import random
import sys

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

class CIFAR_10_CNN:
    def __init__(self):
        sys.path.append('../cs231n')
        from data_utils import load_CIFAR10
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
        self.training_data = X_train
        self.training_labels = y_train
        self.test_data = X_test
        self.test_labels = y_test

def main():
    cnn = CIFAR_10_CNN()

if __name__ == "__main__":
    main()