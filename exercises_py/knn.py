#!/usr/bin/env python
# coding: utf-8

# # k-Nearest Neighbor (kNN)
# 
# *Work through the KNN classifier to gain an understanding of how the classifier works, how it is implemented and how the hyperparameters affect its performance. Use the results in your exam report.*
# 
# The kNN classifier consists of two stages:
# 
# - During training, the classifier takes the training data and simply remembers it
# - During testing, kNN classifies every test image by comparing to all training images and transfering the labels of the k most similar training examples
# - The value of k is a hyperparameter and is cross-validated
# 
# In this exercise you will get an understanding of the basic Image Classification pipeline, cross-validation, and learn about efficient, vectorized code.

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


# Run some setup code for this notebook.

import sys
sys.path.append('../cs231n')
import random
import numpy as np
from data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import time

# # This is a bit of magic to make matplotlib figures appear inline in the notebook
# # rather than in a new window.
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# # Some more magic so that the notebook will reload external python modules;
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# # In[ ]:


# # Load the raw CIFAR-10 data.
cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'

 # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train, y_train
    del X_test, y_test
    print('Clear previously loaded data.')
except:
    pass

X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# # As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# # In[ ]:


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8'))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()


# # In[ ]:


# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
print(X_train.shape, X_test.shape)

# Reshape the image data into rows (flattening)
X_train = np.reshape(X_train, (X_train.shape[0], -1)) #order of -1 tells Numpy to figure the remaining of the shape out by itself (magic!)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# # In[ ]:


from classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
# classifier = KNearestNeighbor()
# t1 = time.time()
# classifier.train(X_train, y_train)
# t2 = time.time()
# print(t2-t1)


# We would now like to classify the test data with the kNN classifier. Recall that we can break down this process into two steps: 
# 
# 1. First we must compute the distances between all test examples and all train examples. 
# 2. Given these distances, for each test example we find the k nearest examples and have them vote for the label
# 
# Lets begin with computing the distance matrix between all training and test examples. For example, if there are **Ntr** training examples and **Nte** test examples, this stage should result in a **Nte x Ntr** matrix where each element (i,j) is the distance between the i-th test and j-th train example.
# 
# The distance calculation is implemented in the function `compute_distances_two_loops` in the file `cs231n/classifiers/k_nearest_neighbor.py`. This implementation uses a (very inefficient) double loop over all pairs of (test, train) examples and computes the distance matrix one element at a time.

# In[ ]:


# Test the implementation:
# t1 = time.time()
# dists = classifier.compute_distances_two_loops(X_test)
# t2 = time.time()
# print("distances shape after compute", dists.shape)
# print("time to compute distance 2 loops", t2-t1)


# # **Inline Questions Part 1** 
# # 
# # We would like for you to record the time it takes for the kNN-classifier to train and classify the test data (with compute_distances_two_loops) with a couple different training- and test-data sizes. (You will need to reload the data in one of the cells above and then change num_training and num_test before subsampling)
# # 
# # - How does the size of the training set affect the training time? it doesen't really. lazy learning means training time isn't really present
# # - How does the size of the test set & training set affect the test time? linearly (n*m)
# # - Can you explain this with your theoretical knowledge about the kNN-classifier? 
# # 
# # $\color{green}{\textit Your Answer:}$ *fill this in.*


# # In[ ]:


# # We can visualize the distance matrix.
#plt.imshow(dists, interpolation='none')
#plt.show()


# # **Inline Questions Part 2** 
# # 
# # Notice the structured patterns in the distance matrix, where some rows or columns are visible brighter. (Note that with the default color scheme black indicates low distances while white indicates high distances.)
# # 
# # - What is on the x-axis? 
# # - What is on the y-axis?
# # - What in the data is the cause behind the distinctly bright rows?
# # - What causes the distinctly bright columns?
# # - What can you use this information for?
# # 
# # $\color{green}{\textit Your Answer:}$ *fill this in.*
# # 
# # 

# # In[ ]:


# Take a look at the implementation of the function 'predict_labels' in the file 'cs231n/classifiers/k_nearest_neighbor.py' and run the code below:
# We use k = 1 (which is Nearest Neighbor).
# y_test_pred = classifier.predict_labels(dists, k=1)

# # Compute and print the fraction of correctly predicted examples
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# # You should expect to see approximately `27%` accuracy. Now lets try out a larger `k`, say `k = 5`:

# # In[ ]:


# y_test_pred = classifier.predict_labels(dists, k=5)
# num_correct = np.sum(y_test_pred == y_test)
# accuracy = float(num_correct) / num_test
# print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# # You should expect to see a slightly better performance than with `k = 1`.
# # 
# # **Inline Questions Part 3**
# # 
# # - Why does the performance change with different values of k?
# # - What is happening inside the 'predict_labels' function?
# # 
# # $\color{green}{\textit Your Answer:}$ *fill this in.*
# # 

# # In[ ]:


# # Now lets speed up distance matrix computation by using partial vectorization
# # with one loop. This is implemented in the function 'compute_distances_one_loop'. 
# # Run the code below:
# dists_one = classifier.compute_distances_one_loop(X_test)

# # To ensure that our vectorized implementation is correct, we make sure that it
# # agrees with the naive implementation. There are many ways to decide whether
# # two matrices are similar; one of the simplest is the Frobenius norm. In case
# # you haven't seen it before, the Frobenius norm of two matrices is the square
# # root of the squared sum of differences of all elements; in other words, reshape
# # the matrices into vectors and compute the Euclidean distance between them.
# difference = np.linalg.norm(dists - dists_one, ord='fro')
# print('One loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')


# # In[ ]:


# We can speed up the distance matrix computation even more by implementing a fully vectorized version. This is done inside 'compute_distances_no_loops'. Check it out!
# Run the code:
# dists_two = classifier.compute_distances_no_loops(X_test)

# # check that the distance matrix agrees with the one we computed before:
# difference = np.linalg.norm(dists - dists_two, ord='fro')
# print('No loop difference was: %f' % (difference, ))
# if difference < 0.001:
#     print('Good! The distance matrices are the same')
# else:
#     print('Uh-oh! The distance matrices are different')


# In[ ]:


# Let's compare how fast the implementations are
# def time_function(f, *args):
#     """
#     Call a function f with args and return the time (in seconds) that it took to execute.
#     """
#     import time
#     tic = time.time()
#     f(*args)
#     toc = time.time()
#     return toc - tic

# two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
# print('Two loop version took %f seconds' % two_loop_time)

# one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
# print('One loop version took %f seconds' % one_loop_time)

# no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
# print('No loop version took %f seconds' % no_loop_time)

# You should see significantly faster performance with the fully vectorized implementation!

# NOTE: depending on what machine you're using, you might not see a speedup when you go from two loops to one loop, and might even see a slow-down.


# ### Cross-validation
# 
# We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily. We will now determine the best value of this hyperparameter with cross-validation.
# 
# TODO: 
# - Choose a set of values for k to use in the cross-validation test (e.g. [1, 3, 5, 8, 10, 12, 15, 20, 50, 100])
# - Good practice is to start with a wide range of values for your hyperparameter, and gradually narrow it down.

# In[ ]:


num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100] #TODO: Choose a set of values for k to use in the cross-validation test

X_train_folds = []
y_train_folds = []

################################################################################
# What we do:                                                                  #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
################################################################################

X_train_folds = np.array_split(X_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# What you need to do:                                                         #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################

for k_val in k_choices:
    k_to_accuracies[k_val] = []
    for i in range(num_folds):
        classifier =   KNearestNeighbor()# TODO !!! You need to reinitialize our classifier for each fold, so that it does not carry any "memory" over from previous folds
        training_data = np.concatenate([ fold for j, fold in enumerate(X_train_folds) if i != j ])  # Our training folds are all the folds except i
        training_data_labels = np.concatenate([ fold for j, fold in enumerate(y_train_folds) if i != j ]) # Same for our labels
          # TODO !!!  Now you need to train the classifier on the training data and labels
        classifier.train(training_data, training_data_labels)
        y_test_pred = classifier.predict(X_train_folds[i], k=k_val)  # The only fold used to evaluate the performance is fold i
        
        #  TODO !!! Compute the proportion of correctly predicted examples (accuracy) and append it to the dictionary "k_to_accuracies"
        accuracy = np.sum(y_test_pred == y_train_folds[i])/len(y_test_pred)
        k_to_accuracies[k_val].append(accuracy)

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))


# In[ ]:

max_k = 0
max_acc = 0

# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    for acc in accuracies:
        if acc > max_acc:
            max_k = k
            max_acc = acc
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
# plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
# plt.title('Cross-validation on k')
# plt.xlabel('k')
# plt.ylabel('Cross-validation accuracy')
# plt.show()


# In[ ]:


# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.

best_k = max_k #  TODO !!!  Set the value of k which performed best in your cross-validation test

# Reshape the image data into rows (flattening)
X_train = np.reshape(X_train, (X_train.shape[0], -1)) #order of -1 tells Numpy to figure the remaining of the shape out by itself (magic!)
X_test = np.reshape(X_test, (X_test.shape[0], -1))
#X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

num_test = X_test.shape[0]

# TODO !!! Initialize the KNN classifier, train it on all of the training data (not just the sub-set) and predict labels of the test set
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
                            ## train 
y_test_pred = classifier.predict(X_test, k=max_k)

# TODO !!!  Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = np.sum(y_test_pred == y_test) / len(y_test_pred)
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# **Inline Questions Part 4**
# 
# - How did you structure your search for the best value of k? tried a bunch of different k value in cross-validation and selected the one with max acc
# - What did you find? Comment on your graph.
# - What is your explanation for the performance of really low values of k and really high values of k (e.g. k = number of training samples)? predictably bad. low k captures
# no nuances, high k draws in too many
# - What happens if the k-nearest neighbors contain an equal number of two or more different classes? different from implementation to implementation.
# if we use threshold we can avoid a classification, we can also pick at random, we can also choose k to be a non equal number to limit the chance
# - Why do we mainly care about the validation- / test accuracy? 
# - What would happen if you tried to use kNN with k=1 to classify the training data?
# 
# $\color{green}{\textit Your Answer:}$ *fill this in.*
# 
# 
