# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf
import pandas as pd

import numpy as np

import os

from predict import predict
from preprocess import preprocess


def _cost(predictions, observations):
    return tf.reduce_mean(predictions - observations * tf.log(predictions + 1e-10) + tf.lgamma(observations + 1.0))


def _training_step(cost):
    return tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


def _build_graph(image_width, image_height):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # turn off tensorflow warning messages
    tf.reset_default_graph()
    tf.set_random_seed(1)

    # dimensions of X are [number_examples, image_width, image_height, number_channels]:
    X = tf.placeholder(tf.float32, [None, image_width, image_height, 1])
    predictions = predict(X)

    y = tf.placeholder(tf.float32, shape=[None, 1])

    cost = _cost(predictions, y)
    training_step = _training_step(cost)

    return {'X': X, 'y': y, 'predictions': predictions, 'cost': cost, 'training_step': training_step}


def train(data, number_epochs, logging_interval, print_to_console):
    images, depths = preprocess(data)

    _, image_width, image_height, _ = images.shape
    graph = _build_graph(image_width, image_height)

    log = []
    predicted_depths = None
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in range(number_epochs):
            _, cost, predicted_depths = session.run(
                [graph['training_step'], graph['cost'], graph['predictions']],
                feed_dict={graph['X']: images, graph['y']: depths})
            number_depths = 3
            if epoch % logging_interval == 0:
                if print_to_console:
                    print('epoch:', epoch,
                          'cost:', cost,
                          'observed_depths:', depths[:number_depths, 0],
                          'predicted_depths:', predicted_depths[:number_depths, 0])
                # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
                # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
                log.append({'epoch': epoch, 'cost': cost})

    log = pd.DataFrame(log)
    data['predicted_depth'] = predicted_depths

    print('number of trainable parameters:',
          np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    print(tf.trainable_variables())

    return data, log
