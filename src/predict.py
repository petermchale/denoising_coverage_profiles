# coding=utf-8

from __future__ import print_function

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import tensorflow as tf

import numpy as np


def _convolution_activation_pooling(inputs, num_filters, filter_shape, pool_shape):
    num_input_channels = inputs.shape.as_list()[-1]

    # initialise weights (of filter) and biases
    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    weights = tf.get_variable('weights',
                              shape=[filter_shape[0], filter_shape[1], num_input_channels, num_filters],
                              initializer=tf.truncated_normal_initializer(stddev=0.05))

    biases = tf.get_variable('biases',
                             shape=[num_filters],
                             initializer=tf.zeros_initializer())

    # set up the convolutional layer operation
    outputs = tf.nn.conv2d(inputs, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    # The argument [1, 1, 1, 1] is the strides parameter: we want the filter to move in steps of 1 in both the x and
    # y directions (or height and width directions). This information is conveyed in the strides[1] and strides[2]
    # values. The first and last values of strides are always equal to 1, if they were not, we would be moving the
    # filter between training samples or between channels, which we don’t want to do.

    # The final parameter is the padding. This padding is to avoid the fact that, when traversing a (x,y) sized image
    # or input with a convolutional filter of size (n,m), with a stride of 1 the output would be (x-n+1,y-m+1).  So
    # in this case, without padding, the output size would be "smaller".  We want to keep the sizes of the outputs easy
    # to track, so we chose the “SAME” option as the padding so we keep the same size.

    # add the biases
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    outputs += biases

    # apply a ReLU non-linear activation
    outputs = tf.nn.relu(outputs)

    # now perform max pooling
    outputs = tf.nn.max_pool(outputs,
                             ksize=[1, pool_shape[0], pool_shape[1], 1],
                             strides=[1, 1, 2, 1],
                             padding='SAME')

    # we return an object, which is actually a sub-graph of its own, containing all the operations and
    # weight variables within it
    return outputs


def _dense(inputs, number_output_nodes):
    number_input_nodes = inputs.shape.as_list()[-1]

    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    weights = tf.get_variable(name='weights',
                              shape=(number_input_nodes, number_output_nodes),
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases',
                             shape=number_output_nodes,
                             initializer=tf.zeros_initializer())

    # we return an object, which is actually a sub-graph of its own, containing all the operations and
    # weight variables within it
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    return tf.matmul(inputs, weights) + biases


def _flatten(layer):
    _, image_width, image_height, number_filters = layer.shape.as_list()

    return tf.reshape(layer, [-1, image_width * image_height * number_filters])


def _number_trainable_parameters():
    return np.sum([np.prod(variable.get_shape().as_list()) for variable in tf.trainable_variables()])


def predict(X):
    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    with tf.variable_scope('conv1'):
        conv1 = _convolution_activation_pooling(X, num_filters=25, filter_shape=[4, 8], pool_shape=[1, 2])

    with tf.variable_scope('conv2'):
        conv2 = _convolution_activation_pooling(conv1, num_filters=50, filter_shape=[4, 8], pool_shape=[1, 2])

    flattened = _flatten(conv2)

    with tf.variable_scope('dense'):
        dense = tf.nn.relu(_dense(flattened, number_output_nodes=500))

    with tf.variable_scope('output'):
        predictions = tf.exp(_dense(dense, number_output_nodes=1), name='predictions')

    print('number of trainable parameters:', _number_trainable_parameters())

    return predictions
