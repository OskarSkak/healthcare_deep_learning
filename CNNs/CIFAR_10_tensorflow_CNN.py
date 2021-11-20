import numpy as np
import pyreadr
import random
import sys

from matplotlib import pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical

from sklearn.utils import shuffle

class CIFAR_10_CNN:
    def __init__(self):
        sys.path.append('../cs231n')
        from data_utils import load_CIFAR10
        cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
        
        X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
        #Prepare data
        #Make labels categorical
        self.trainX = X_train
        self.trainY = to_categorical(y_train)
        self.testX = X_test
        self.testY = to_categorical(y_test)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        #normalize
        train_norm = self.trainX.astype('float32')
        test_norm = self.testX.astype('float32')
        self.train_norm = train_norm /255.0
        self.test_norm = test_norm / 255.0

    def print_shapes(self):
        print("training data/labels", self.training_data.shape, self.training_labels.shape)
        print("test data/labels", self.test_data.shape, self.test_labels.shape)
        #training data/labels (50000, 32, 32, 3) (50000,)
        #test data/labels (10000, 32, 32, 3) (10000,)

    # define cnn model
    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def print_rand_img(self):
        cols = 8
        rows = 4

        fig = plt.figure(figsize=(2 * cols, 2*rows))

        for col in range(cols):
            for row in range(rows):
                rand = np.random.randint(0, len(self.training_labels))
                ax = fig.add_subplot(rows, cols, col * rows + row + 1)
                ax.grid(b=False)
                ax.axis("off")
                ax.imshow(self.training_data[rand, :])
                ax.set_title("")#self.classes[self.training_labels[rand][0]])
        plt.show()

def main():
    cnn = CIFAR_10_CNN()
    cnn.print_shapes()
    


if __name__ == "__main__":
    main()