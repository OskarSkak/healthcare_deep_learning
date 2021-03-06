#!/usr/bin/env python
# coding: utf-8

# # Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.

# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive')


# In[ ]:


# A bit of setup

import sys
sys.path.append('../cs231n')
import numpy as np
import matplotlib.pyplot as plt
from classifiers.neural_net import TwoLayerNet

# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# # for auto-reloading external modules
# # see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


# We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop our implementation.

# In[ ]:


# Create a small net and some toy data to check our implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()


# **Inline Questions Part 1**
# 
# Sometimes when training Neural Networks and Convolutional Neural Networks, it can be useful to set the random seed to be able to reproduce results.
# 
# - Which part of the initialization of a neural network involves random processes? (If you don't remember, you can check the implementation of 'TwoLayerNet.__init__()')
# - Which part of the training of a neural network involves random processes? (If you don't remember, you can check the implementation of 'TwoLayerNet.train()')
# 
# $\color{green}{\textit Your Answer:}$ *fill this in*  

# # Forward pass: compute scores
# Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. This function is very similar to the loss functions that were written for the SVM and Softmax exercises: It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 

# In[ ]:


scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. I get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))


# # Forward pass: compute loss
# Now check the loss function is implemented correctly.

# In[ ]:


loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))


# # Backward pass
# In the backward pass,the gradient of the loss with respect to the variables `W1`, `b1`, `W2`, and `b2` are computed. Check that the implementation of the backward pass is correct using a numeric gradient check:

# In[ ]:


from gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))


# # Train the network
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` which implements the training procedure. This should be very similar to the training procedure that was used for the SVM and Softmax classifiers. 
# Also check out the implementation of `TwoLayerNet.predict`, which is used periodically during the training process to perform predictions to keep track of accuracy over time while the network trains.
# 
# Run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.02.

# In[ ]:


net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# # plot the loss history
# plt.plot(stats['loss_history'])
# plt.xlabel('iteration')
# plt.ylabel('training loss')
# plt.title('Training Loss history')
# plt.show()


# **Inline Questions Part 2**
# 
# - Is the loss at the beginning of the training process close to what you expect it to be? What do you expect it to be?
# - From looking at the graph, do you think the choice of learning rate is good or bad?
# 
# $\color{green}{\textit Your Answer:}$ *fill this in*  

# # Load the data
# Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.

# In[ ]:


from data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = '../cs231n/datasets/cifar-10-batches-py'
    
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# # Train a network
# To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

# In[ ]:


input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=False)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)


# **Inline Questions Part 3**
# 
# - What is the hidden_size hyperparameter used for?
# - How many neurons are there in the last layer of the neural network? Why?
# - Is the loss at the beginning of training close to what you expect it to be? How do you calculate the expected loss?
# - Why does it make sense to decay the learning rate during the training process?
# 
# $\color{green}{\textit Your Answer:}$ *fill this in*  

# # Debug the training
# With the default parameters we provided above, you should get a validation accuracy of about 0.28-0.29 on the validation set. This isn't very good.
# 
# One strategy for getting insight into what's wrong is to plot the loss function and the accuracies on the training and validation sets during optimization.
# 
# Another strategy is to visualize the weights that were learned in the first layer of the network. In most neural networks trained on visual data, the first layer weights typically show some visible structure when visualized.

# In[ ]:


# Plot the loss function and train / validation accuracies

# plt.subplot(2, 1, 1)
# plt.plot(stats['loss_history'])
# plt.title('Loss history')
# plt.xlabel('Iteration')
# plt.ylabel('Loss')

# plt.subplot(2, 1, 2)
# plt.plot(stats['train_acc_history'], label='train')
# plt.plot(stats['val_acc_history'], label='val')
# plt.title('Classification accuracy history')
# plt.xlabel('Epochs')
# plt.ylabel('Classification accuracy')
# plt.legend()
# plt.show()


# In[ ]:


from vis_utils import visualize_grid

# Visualize the weights of the network

# def show_net_weights(net):
#     W1 = net.params['W1']
#     W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
#     plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
#     plt.gca().axis('off')
#     plt.show()

# show_net_weights(net)


# # Tune your hyperparameters
# 
# **What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.
# 
# **Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.
# 
# **Approximate results**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.
# 
# **Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can (52% could serve as a reference), with a fully-connected Neural Network.
# 

# **Explain your hyperparameter tuning process below.**
# 
# $\color{Green}{\textit Your Answer:}$

# In[ ]:


best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################

best_valacc = -1

hidden_sizes = [10, 30, 80, 175, 300]  # TODO: Select a set of hidden_sizes

hist_val_acc = []
hist_hs = []
hist_lr = []
hist_lr_dec = []
hist_reg_str = []
hist_bs = []
hist_iters = []

#input size: 3072
#output size: 10
#Soft rule: Masters (1993): sqrt(n*m) -> sqrt(10*3072) =~ 175
#   obviously not perfect, but starting point, we will work around it
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # TODO: Select a set of learning_rates -> just following rule of thumb here
learning_rate_decays = [0.95]
regularization_strengths = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # # TODO: Select a set of regularization strengths: Same deal as learning rates (RoT)
batch_sizes = [200]  # TODO: You can also try to see how the batch size can influence training
num_iters = [1500]  # During hyperparameter tuning it is common to train for a shorter amount of time
for hs in hidden_sizes:
    for lr in learning_rates:
        for lr_dec in learning_rate_decays:
            for reg_str in regularization_strengths:
                for bs in batch_sizes:
                    for iters in num_iters:
                        input_size = 32 * 32 * 3
                        hidden_size = hs
                        num_classes = 10
                        net = TwoLayerNet(input_size, hidden_size, num_classes)

                        # Train the network
                        stats = net.train(X_train, y_train, X_val, y_val,
                                    num_iters=iters, batch_size=bs,
                                    learning_rate=lr, learning_rate_decay=lr_dec,
                                    reg=reg_str, verbose=False)

                        # Predict on the validation set
                        val_acc = (net.predict(X_val) == y_val).mean()
                        
                        # print("Val Acc.: " + str(val_acc) + " - hyperparams: " + "hs: " + str(hs) + " lr: " + str(lr) 
                        #       + " lr dec: " + str(lr_dec) + " reg_str: " + str(reg_str) + " batch size: " + str(bs)
                        #       + " iterations: " + str(iters))

                        hist_val_acc.append(val_acc)
                        hist_hs.append(hs)
                        hist_lr.append(lr)
                        hist_lr_dec.append(lr_dec)
                        hist_reg_str.append(reg_str)
                        hist_bs.append(bs)
                        hist_iters.append(iters)
                        
                        if val_acc > best_valacc:
                            best_valacc = val_acc
                            best_net = net
                        
print("Best model reached a validation accuracy of: " + str(best_valacc))


# In[ ]:


# visualize the weights of the best network
#show_net_weights(best_net)


# # Run on the test set
# When you are done experimenting, you should evaluate your final trained network on the test set; you should be able to get above 48%.

# In[ ]:


test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)


# **Inline Questions Part 4**
# 
# Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy (there is overfitting). In what ways can we decrease this gap? Select all that apply.
# 
# 1. Train on a larger dataset.
# 2. Add more hidden units.
# 3. Increase the regularization strength.
# 4. Increase the learning rate.
# 5. None of the above.
# 
# $\color{Green}{\textit Your Answer:}$
# 
# You may also see that the testing accuracy is lower than the validation accuracy. Why may this be?
# 
# $\color{green}{\textit Your Explanation:}$
# 
# 

def print_res(msg, vals):
    print(msg, end='')
    for x in vals:
        print(x, ", ", end='')

    print("\n******************************************************************************************************************************************")

print_res("accuracy", hist_val_acc)
print_res("hs", hist_hs)
print_res("lr", hist_lr)
print_res("lr_dec", hist_lr_dec)
print_res("reg_str", hist_reg_str)
print_res("bs", hist_bs)
print_res("iters", hist_iters)


"""hist_val_acc = []
hist_hs = []
hist_lr = []
hist_lr_dec = []
hist_reg_str = []
hist_bs = []
hist_iters = []"""


#All the results: 
# Test accuracy:  0.476
# accuracy0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.399 , 0.379 , 0.363 , 0.379 , 0.377 , 0.311 , 0.319 , 0.298 , 0.302 , 0.322 , 0.196 , 0.215 , 0.189 , 0.18 , 0.178 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.44 , 0.426 , 0.421 , 0.425 , 0.421 , 0.321 , 0.322 , 0.332 , 0.325 , 0.317 , 0.213 , 0.204 , 0.216 , 0.22 , 0.225 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.484 , 0.443 , 0.453 , 0.452 , 0.449 , 0.338 , 0.33 , 0.335 , 0.341 , 0.335 , 0.256 , 0.227 , 0.214 , 0.233 , 0.207 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.488 , 0.436 , 0.462 , 0.477 , 0.465 , 0.334 , 0.344 , 0.335 , 0.337 , 0.345 , 0.225 , 0.214 , 0.206 , 0.24 , 0.248 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.087 , 0.464 , 0.448 , 0.463 , 0.503 , 0.458 , 0.345 , 0.354 , 0.346 , 0.351 , 0.351 , 0.219 , 0.207 , 0.246 , 0.256 , 0.222 ,
# ******************************************************************************************************************************************
# hs10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 10 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 30 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 80 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 175 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 ,
# ******************************************************************************************************************************************
# lr0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 0.1 , 0.1 , 0.1 , 0.1 , 0.1 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.001 , 0.001 , 0.001 , 0.001 , 0.001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 0.0001 , 1e-05 , 1e-05 , 1e-05 , 1e-05 , 1e-05 ,
# ******************************************************************************************************************************************
# lr_dec0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 , 0.95 ,
# ******************************************************************************************************************************************
# reg_str0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 , 0.1 , 0.01 , 0.001 , 0.0001 , 1e-05 ,
# ******************************************************************************************************************************************
# bs200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 , 200 ,
# ******************************************************************************************************************************************
# iters1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 , 1500 ,
# ******************************************************************************************************************************************