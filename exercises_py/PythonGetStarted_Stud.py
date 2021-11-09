#!/usr/bin/env python
# coding: utf-8

# # Assignment: Get Started in Python & common libraries
# Complete this notebook for lecture 2.
# 
# Then we will discuss it in groups & in class.

# In[ ]:


import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2 as cv

print("Python version:")
print(sys.version)
print("\nNumpy version:")
print(np.__version__)
print("\nMatplotlib version:")
print(mpl.__version__)
print("\nOpenCV version:")
print(cv.__version__)


# # In[ ]:


# # Create a list of 5 strings, where each string is some object that you can see near you
# list_of_things = []


# # In[ ]:


# # Make a for loop that iterates over every element of your list and prints it out


# # In[ ]:


# # Use np.arange() to create a numpy array that contains all integers between 0 and 100 (including 100)
# # Then write a for loop that iterates over every element and prints the number only if the number is even.
# # Hint: Use an if-statement and the modulus operator (%)
# array_of_ints = 


# # In[ ]:


# # We have given you a python list containing integers between 0 and 10.
# # Convert this list to a numpy array where every integer has been doubled (do not use a for loop for this) and print out the contents of the array.
# just_a_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# # In[ ]:


# # We have given you a numpy array containing 50 random integers in the range between 0 and 1000.
# # We will use this in the next few cells.
# rand_ints = np.random.randint(low=0, high=1000, size=50)
# print(rand_ints)


# # In[ ]:


# # This code is filled with errors. There is one error on every line except the first and the last.
# # Hint: Remember that wrong indentations can give unexpected errors in Python.
# i = 0
# while i <= len(rand_ints):
#   if rand_ints[i] < 100
#     rand_ints(i) *= 5
# i+= 1

# print(rand_ints)


# # In[ ]:


# # Find a numpy method that can sort rand_ints for you in ascending order.
# # Print the beautifully sorted array.

# sorted_array = 


# # In[ ]:


# # Find numpy methods that can give you the smallest and largest integers in your array of integers (rand_ints or sorted_array).
# # Use these values to normalize your array of integers to be in the range between 0 and 1.
# # Hint: https://www.codecademy.com/articles/normalization#:~:text=Min%2Dmax%20normalization%20is%20one,decimal%20between%200%20and%201.

# max_val = 
# min_val = 
# normalized = 

# # Check your code works by printing the normalized array
# # Question! Which data type are the values in your normalized array?


# # In[ ]:


# # Find a numpy method for converting your normalized array to data type "float16"

# normalized = 


# # In[ ]:


# # Make a python function that can take as argument a python list of integers. It must then do the following:
# # 1) Convert the python list to a numpy array
# # 2) Sort the array in ascending order
# # 3) Normalize the array to the range between -0.5 and 0.5
# # 4) Convert the array to data type "float32"
# # 5) Print the normalized array, its data type, and return it

# def normalizer():
  
#   return 

# l = list(range(0, 255))
# nor = normalizer(l)


# # In[ ]:


# # Use the plot() method from matplotlib.pyplot to visualize your normalized array from the cell above.


# # In[ ]:


# # We are mounting our drive to enable file exploring
# from google.colab import drive
# drive.mount('/content/drive')


# # In[ ]:


# # Find an image from anywhere, put it in Google Drive, load it and show it using OpenCV functions
# # In Colab we have to use cv2_imshow() instead of cv.imshow()
# from google.colab.patches import cv2_imshow


# # In[ ]:


# # Normalize the image to be in the 0-1 range
# # Use print statements to check the min/max pixel values before and after normalization


# # In[ ]:


# # Find a way to in a single line of code adjust the brightness and/or contrast of the image (without iterating through pixels with for-loops)
# # show it to verify that it works.
# alpha = 1.  # scaling factor
# beta = 50  # shift factor


# # In[ ]:


# # Using OpenCV functions, find a way to horizontally and vertically flip your image
# # show it to verify that it works.


# # In[ ]:


# # Find a way to extract a patch of size 100x100 exactly from the center of your image and show it to verify that it works.
# # Hint: You can get the width and height of your image (numpy array) by img.shape[0] and img.shape[1]
# patch_width = 100
# patch_height = 100
# extract_position = []


# # In[ ]:


# # Make a function that is able to extract N patches of size MxM from random locations in your image and show them to verify that it works.
# import random
# n = 5
# m = 50

# def extract_patches(image, num_extract:int, size:int):
  

