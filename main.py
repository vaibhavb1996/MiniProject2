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

def create_flatten_layer(layer):
	#retrieve the shape of the layer
	layer_shape = layer.get_shape()

	#Number of features shall be height*width*channels
	num_features = layer_shape[1:4].num_elements()

	#Now we flatten the layer so we shall have to reshape to num_features
	layer = tf.reshape(layer, [-1, num_features])

	return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu = True):
	#Defining trainable weights and biases
	weights = create_weights(shape = [num_inputs, num_outputs])
	biases = create_biases(num_outputs)

	# Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	return layer

layer_conv1 = create_convolutional_layer(input = x, num_input_channels = num_channels, conv_filter_size = filter_size_conv1, num_filters = num_filters_conv1)
layer_conv2 = create_convolutional_layer(input = layer_conv1, num_input_channels = num_filters_conv1, conv_filter_size = filter_size_conv2, num_filters = num_filters_conv2)
layer_conv3 = create_convolutional_layer(input = layer_conv2, num_input_channels = num_filters_conv2, conv_filter_size = filter_size_conv3, num_filters = num_filters_conv3)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input = layer_flat, num_inputs = layer_flat.get_shape()[1:4].num_elements(),num_outputs = fc_layer_size, use_relu = True)
layer_fc2 = create_fc_layer(input = layer_fc1, num_inputs = fc_layer_size, num_outputs = num_classes, use_relu = False)


