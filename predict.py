# coding=utf-8

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
import tensorflow as tf


def _convolution_activation_pooling(inputs, num_filters, filter_shape, pool_shape):
    num_input_channels = inputs.shape.as_list()[-1]

    # initialise weights (of filter) and biases
    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    weights = tf.get_variable('weights',
                              shape=[filter_shape[0], filter_shape[1], num_input_channels, num_filters],
                              initializer=tf.truncated_normal_initializer(stddev=0.05))
    tf.summary.histogram('weights', weights)

    biases = tf.get_variable('biases',
                             shape=[num_filters],
                             initializer=tf.zeros_initializer())
    tf.summary.histogram('biases', biases)

    # set up the convolutional layer operation
    # https://github.com/gifford-lab/Keras-genomics/blob/master/example/model.py
    # https://github.com/gifford-lab/Keras-genomics/issues/1#issuecomment-270767629
    outputs = tf.nn.conv2d(inputs, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    # We want the filter to move in steps of 1 in both the
    # height and width directions, which is conveyed in the strides[1] and strides[2]
    # values, respectively. The first and last values of "strides" are always equal to 1, if they were not,
    # we would be skipping examples and/or input channels, respectively [see the "data_format" parameter of tf.nn.conv2d]

    # The final parameter is the padding. This padding is to avoid the fact that, when traversing a (h,w) sized image
    # or input with a convolutional filter of size (n,m), with a stride of 1 the output would be (h-n+1,w-m+1).  So
    # in this case, without padding, the output size would be "smaller".  Choose
    # the “SAME” option for the padding to retain the image size after convolution.

    # add the biases
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    outputs += biases
    tf.summary.histogram('neuron_input', outputs)

    # apply a ReLU non-linear activation
    outputs = tf.nn.relu(outputs)

    # now perform max pooling
    outputs = tf.nn.max_pool(outputs,
                             ksize=[1, pool_shape[0], pool_shape[1], 1],
                             strides=[1, 1, 2, 1],
                             padding='SAME')
    # The kernel size ksize would be, e.g., [1, 2, 2, 1]
    # if you have a 2x2 window over which you take the maximum.
    # On the batch dimension and the channel dimension, ksize is 1,
    # because we neither want to take the maximum over multiple examples, nor over multiple channels.
    # That is: the same pooling procedure is applied to all channels, independently

    # https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t

    # we return an object, which is actually a sub-graph of its own, containing all the operations and
    # weight variables within it
    return outputs


def _dense(inputs, number_output_nodes):
    number_input_nodes = inputs.shape.as_list()[-1]

    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    weights = tf.get_variable(name='weights',
                              shape=(number_input_nodes, number_output_nodes),
                              initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram('weights', weights)

    biases = tf.get_variable(name='biases',
                             shape=number_output_nodes,
                             initializer=tf.zeros_initializer())
    tf.summary.histogram('biases', biases)

    # we return an object, which is actually a sub-graph of its own, containing all the operations and
    # weight variables within it:
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    outputs = tf.matmul(inputs, weights) + biases # dimensions will be: [number_examples, number_output_nodes]
    tf.summary.histogram('neuron_input', outputs)

    return outputs


def _flatten(layer):
    _, image_height, image_width, number_filters = layer.shape.as_list()

    # first dimension of return value will be: number_examples
    return tf.reshape(layer, [-1, image_height * image_width * number_filters])


def predict(X):
    # https://www.tensorflow.org/versions/r1.1/programmers_guide/variable_scope
    with tf.variable_scope('conv1_layer'):
        conv1 = _convolution_activation_pooling(X, num_filters=25, filter_shape=[4, 8], pool_shape=[1, 2])

    # A potential problem: with padding=same and filter height=4,
    # the second conv layer will pad the output from the 1st layer, increasing its height dimension from 1 to 4,
    # which means that 3/4 of the filter values get multiplied by zero all the time, which is wasteful.
    with tf.variable_scope('conv2_layer'):
        conv2 = _convolution_activation_pooling(conv1, num_filters=50, filter_shape=[4, 8], pool_shape=[1, 2])

    flattened = _flatten(conv2)

    with tf.variable_scope('dense_layer'):
        dense = tf.nn.relu(_dense(flattened, number_output_nodes=500))

    with tf.variable_scope('output_layer'):
        predictions = tf.exp(_dense(dense, number_output_nodes=1), name='predictions')

    return predictions
