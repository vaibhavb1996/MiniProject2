# Copyright 2018 Vaibhav Bansal vbansal@bu.edu
import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

batch_size = 32

# Preparing input data
classes = os.listdir('training_data')
num_classes = len(classes)

# validation data segmentation
validation_size = 0.15
img_size = 128
num_channels = 3
train_path = 'training_data'

# All data is loaded into memory using OpenCV
data = dataset.read_trai_sets(train_path, img_size, classes, validation_size = validation_size)

print("Completed reading input data. Will print a snippet of it now..")
print("No. of files in training set:\t\t{}".format(len(data.train.labels)))
print("No. of files in validation set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape = [None, img_size, img_size, num_channels], name = 'x')

# labels
y_true = tf.placeholder(tf.float32, shape = [None, num_classes], name = 'y_true')
y_true_cls = tf.argmax(y_true, dimension = 1)

#Nerwork graph parameters
filter_size_conv1 = 3
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

def create_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev = 0.05))

def create_biases(size):
	return tf.Variable(tf.constant(0.05, shape = [size]))

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):

	#We shall define the weights that will be trained using create weights function
	weights = create_weights(shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters])
	#create biases using create biases function
	biases = create_biases(num_filters)

	#creating convolutional laer
	layer = tf.nn.conv2d(input = input, filter = weights, strides = [1, 1, 1, 1], padding = 'SAME')
	layer += biases

	#we shall use max-pooling
	layer = tf.nn.max_pool(value = layer, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

	#output of pooling is fed to relu layer which also acts as the activation function 
	layer = tf.nn.relu(layer)

	return layer