from __future__ import print_function

# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings

warnings.filterwarnings('ignore', message='numpy.dtype size changed')
import tensorflow as tf
import pandas as pd

# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np

from sklearn.model_selection import train_test_split

from predict import predict
from load_preprocess_data import load_data, preprocess
from utility import make_dir
from plot import plot_corrected_depths


def _cost(predictions, observations):
    # https://stackoverflow.com/questions/35094899/tensorflow-operator-overloading
    return tf.reduce_mean(predictions - observations * tf.log(predictions + 1e-10) + tf.lgamma(observations + 1.0),
                          name='cost')


def _training_step(cost):
    return tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)


class Graph(object):

    def __init__(self, image_width, image_height):
        # https://www.tensorflow.org/guide/graphs

        tf.reset_default_graph()
        tf.set_random_seed(1)

        # dimensions of X are [number_examples, image_width, image_height, number_channels]:
        with tf.variable_scope('input'):
            self.X = tf.placeholder(tf.float32, shape=[None, image_width, image_height, 1], name='X')
        self.predictions = predict(self.X)

        with tf.variable_scope('cost'):
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
            self.cost = _cost(self.predictions, self.y)

        with tf.variable_scope('training_step'):
            self.training_step = _training_step(self.cost)

        # https://www.tensorflow.org/guide/summaries_and_tensorboard
        with tf.variable_scope('logging'):
            self.cost_summary = tf.summary.scalar('cost', self.cost)
            tf.summary.histogram('predicted_values', self.predictions)
            # create a merged summary op that invokes all previously attached summary ops:
            self.all_summaries = tf.summary.merge_all()


def _current_time():
    import time
    return time.time()


def load_preprocess_data(bed_file, fasta_file, chromosome_number, region_start, region_end):
    data = load_data(bed_file, fasta_file, chromosome_number, region_start, region_end)

    data_train, data_dev = train_test_split(data, test_size=0.2, random_state=42)

    # noinspection PyTypeChecker
    return (data_train,) + preprocess(data_train) + (data_dev,) + preprocess(data_dev)


def _writer(tensorboard_dir, sub_dir_name, session):
    return tf.summary.FileWriter(os.path.join(tensorboard_dir, sub_dir_name), session.graph)


def mini_batches(images, observed_depths, graph, batch_size=256):
    # shuffle the data:
    # https://stats.stackexchange.com/questions/248048/neural-networks-why-do-we-randomize-the-training-set
    # https://stats.stackexchange.com/questions/245502/shuffling-data-in-the-mini-batch-training-of-neural-network
    number_examples = len(observed_depths)
    permutation = np.random.permutation(number_examples)
    shuffled_images = images[permutation, :, :, :]
    shuffled_observed_depths = observed_depths[permutation, :]

    # Here, I implement sampling without replacement, which converges faster than sampling with replacement:
    # theory: https://stats.stackexchange.com/questions/235844/should-training-samples-randomly-drawn-for-mini-batch-training-neural-nets-be-dr
    # experiment: https://stats.stackexchange.com/questions/242004/why-do-neural-network-researchers-care-about-epochs
    batch_indices = np.arange(batch_size)
    while max(batch_indices) < shuffled_images.shape[0]:
        yield {graph.X: shuffled_images[batch_indices, :, :, :],
               graph.y: shuffled_observed_depths[batch_indices, :]}
        batch_indices += batch_size

    # Omitting the last mini-batch for simplicity.
    # Including it MAY require scaling the learning rate by the size of the mini batch
    # min_index = min(batch_indices)
    # yield {graph.X: images[min_index:, :, :, :],
    #        graph.y: observed_depths[min_index:]}


def train(number_epochs, logging_interval, checkpoint_interval,
          print_to_console, train_data_dir, trained_model_dir):
    # !!! use tf.data API when input data is distributed across multiple machines !!!
    # !!! https://www.youtube.com/watch?v=uIcqeP7MFH0 !!!

    if number_epochs <= max(logging_interval, checkpoint_interval):
        print('logging_interval or checkpoint_interval is too large!')
        import sys
        sys.exit()

    (data_train, images_train, observed_depths_train,
     data_dev, images_dev, observed_depths_dev) = load_preprocess_data(
        bed_file=os.path.join(train_data_dir, 'facnn-example.regions.bed.gz'),
        fasta_file=os.path.join(train_data_dir, 'human_g1k_v37.fasta'),
        chromosome_number='1',
        region_start=0,
        region_end=20000)

    _, image_width, image_height, _ = images_train.shape
    graph = Graph(image_width, image_height)

    log = []

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        tensorboard_dir = os.path.join(trained_model_dir, 'tensorboard')

        if os.path.isdir(tensorboard_dir):
            import shutil
            shutil.rmtree(tensorboard_dir)

        # using a context manager, i.e. the "with" syntax,
        # ensures that the FileWriter buffer is emptied, i.e. data are written to disk
        with _writer(tensorboard_dir, 'train', session) as train_writer, \
                _writer(tensorboard_dir, 'dev', session) as dev_writer:

            start_time = _current_time()
            for epoch in range(number_epochs):

                for mini_batch in mini_batches(images_train, observed_depths_train, graph):
                    session.run(graph.training_step, feed_dict=mini_batch)

                feed_dict_train = {graph.X: images_train, graph.y: observed_depths_train}
                cost_train, predicted_depths_train = session.run(
                    [graph.cost, graph.predictions], feed_dict=feed_dict_train)

                feed_dict_dev = {graph.X: images_dev, graph.y: observed_depths_dev}
                cost_dev, predicted_depths_dev = session.run(
                    [graph.cost, graph.predictions], feed_dict=feed_dict_dev)

                if epoch % logging_interval == 0:
                    if print_to_console:
                        number_depths = 3
                        print('epoch:', epoch,
                              'elapsed time (secs):', _current_time() - start_time,
                              'cost_train:', cost_train,
                              'cost_dev:', cost_dev,
                              'observed_depths_train:', observed_depths_train[:number_depths, 0],
                              'predicted_depths_train:', predicted_depths_train[:number_depths, 0])

                    # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop
                    # https://stackoverflow.com/questions/37009287/using-pandas-append-within-for-loop/37009377
                    log.append({'epoch': epoch, 'cost_train': cost_train, 'cost_dev': cost_dev})

                    # https://www.tensorflow.org/guide/summaries_and_tensorboard
                    train_writer.add_summary(session.run(graph.all_summaries, feed_dict=feed_dict_train), epoch)
                    dev_writer.add_summary(session.run(graph.cost_summary, feed_dict=feed_dict_dev), epoch)

                if epoch % checkpoint_interval == 0:
                    graph_variables = 'graph_variables'
                    graph_variables__trained_model_checkpoint = graph_variables + '/trained_model.ckpt'
                    tf.train.Saver().save(session, os.path.join(trained_model_dir,
                                                                graph_variables__trained_model_checkpoint))
                    checkpoint = {
                        # https://stackabuse.com/tensorflow-save-and-restore-models/
                        # https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/
                        # the second argument of save() is the filename prefix of checkpoint files it creates
                        'prefix': graph_variables__trained_model_checkpoint,
                        'graph_variables_dir': graph_variables}
                    import json
                    with open(os.path.join(trained_model_dir, 'checkpoint.json'), 'w') as fp:
                        json.dump(checkpoint, fp, indent=4)

                    data_train['predicted_depth'] = predicted_depths_train
                    data_dev['predicted_depth'] = predicted_depths_dev
                    pickle(data_train.sort_values('start'),
                           data_dev.sort_values('start'),
                           pd.DataFrame(log), trained_model_dir)


def _generate_file_names(dataframe_dir):
    make_dir(dataframe_dir)
    return (os.path.join(dataframe_dir, 'train.pkl'),
            os.path.join(dataframe_dir, 'dev.pkl'),
            os.path.join(dataframe_dir, 'log.pkl'))


def pickle(data_train, data_dev, log, dataframe_dir):
    data_train_filename, data_dev_filename, log_filename = _generate_file_names(dataframe_dir)
    data_train.to_pickle(data_train_filename)
    data_dev.to_pickle(data_dev_filename)
    log.to_pickle(log_filename)


def unpickle(dataframe_dir):
    data_train_filename, data_dev_filename, log_filename = _generate_file_names(dataframe_dir)
    return (pd.read_pickle(data_train_filename),
            pd.read_pickle(data_dev_filename),
            pd.read_pickle(log_filename))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data')
    parser.add_argument('--trained_model')
    args = parser.parse_args()

    train(number_epochs=10, logging_interval=1, checkpoint_interval=1, print_to_console=True,
          train_data_dir=args.train_data, trained_model_dir=args.trained_model)

    train_data, dev_data, train_log = unpickle(args.trained_model)

    from plot import compute_observed_depth_mean
    observed_depth_mean = compute_observed_depth_mean(train_data)

    plot_corrected_depths(train_data, observed_depth_mean, title='train data')
    plot_corrected_depths(dev_data, observed_depth_mean, title='dev data')


if __name__ == '__main__':
    main()
